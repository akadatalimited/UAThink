; uathink.asm - native x86-64-v3 UA identity MLP (FP32 reference).
; Arch: hashed char 3,4-grams D=1024 -> EmbBag(1024x64, mean) -> FC64x16+ReLU -> FC16x5+softmax.
; The (3,4) range matches Python reference features.py effective behavior
; (range(n_range[0],n_range[1]+1) with n_range=(3,4,5) yields n=3,4).
; Deterministic LCG init, online SGD+momentum, UATH v1 checkpoint, syscalls only.
; CLI: uathink selftest | train <train.tsv> <model.bin> [epochs] | predict <model.bin> [ua...]
;      | eval <model.bin> <eval.tsv> | bench
; TSV: "<0-4>\t<ua>\n" ascii. No Python/PyTorch/MCP/CUDA at runtime.
format ELF64 dynamic 3
entry start

SYS_read=0
SYS_write=1
SYS_open=2
SYS_close=3
SYS_fsync=74
SYS_rename=82
SYS_exit=60
O_RDONLY=0
O_WRONLY=1
O_CREAT=64
O_TRUNC=512
O_EXCL=128
MODE_644=420

D_DIM=1024
D_MASK=1023
E_EMB=64
H_HID=16
C_CLS=5
N_EMB=65536
N_FC1=1024
N_B1=16
N_FC2=80
N_B2=5
PAY_BYTES=266644
MAXROWS=60000
CORPUS_SZ=16777216
MAXUALEN=512

segment readable executable
start:
  mov eax,1
  cpuid
  mov eax,ecx
  and eax,0x18401000
  cmp eax,0x18401000
  jne v3fail
  xor ecx,ecx
  xgetbv
  and eax,6
  cmp eax,6
  jne v3fail
  mov eax,7
  xor ecx,ecx
  cpuid
  mov eax,ebx
  and eax,0x128
  cmp eax,0x128
  jne v3fail
  mov eax,0x80000001
  cpuid
  test ecx,1 shl 5
  jz v3fail
  mov rax,[rsp]
  mov [argc_tmp],rax
  lea rax,[rsp+8]
  mov [argv_tmp],rax
  cmp qword [argc_tmp],2
  jb usage
  mov rax,[argv_tmp]
  mov rsi,[rax+8]
  call dispatch
  mov rax,SYS_exit
  xor edi,edi
  syscall
v3fail:
  mov rax,SYS_write
  mov rdi,2
  lea rsi,[e_v3]
  mov rdx,e_v3_l
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall
usage:
  mov rax,SYS_write
  mov rdi,2
  lea rsi,[m_use]
  mov rdx,m_use_l
  syscall
  mov rax,SYS_exit
  mov edi,2
  syscall

; dispatch on argv[1]; rsi=argv[1] on entry, reload from argv_tmp after each streq (clobbers rsi)
dispatch:
  lea rdi,[s_selftest]
  call streq
  test rax,rax
  jnz cmd_selftest
  mov rax,[argv_tmp]
  mov rsi,[rax+8]
  lea rdi,[s_train]
  call streq
  test rax,rax
  jnz cmd_train
  mov rax,[argv_tmp]
  mov rsi,[rax+8]
  lea rdi,[s_predict]
  call streq
  test rax,rax
  jnz cmd_predict
  mov rax,[argv_tmp]
  mov rsi,[rax+8]
  lea rdi,[s_eval]
  call streq
  test rax,rax
  jnz cmd_eval
  mov rax,[argv_tmp]
  mov rsi,[rax+8]
  lea rdi,[s_bench]
  call streq
  test rax,rax
  jnz cmd_bench
  mov rax,[argv_tmp]
  mov rsi,[rax+8]
  lea rdi,[s_hash]
  call streq
  test rax,rax
  jnz cmd_hash
  ret
; streq rsi=arg rdi=lit -> rax 1/0
streq:
  push rbx
.l:
  mov al,[rsi]
  mov bl,[rdi]
  cmp al,bl
  jne .no
  test al,al
  jz .yes
  inc rsi
  inc rdi
  jmp .l
.yes:
  mov eax,1
  pop rbx
  ret
.no:
  xor eax,eax
  pop rbx
  ret

print_str:
  mov rax,SYS_write
  mov rdi,1
  syscall
  ret
print_u64:
  push rbx
  lea rsi,[numbuf+31]
  mov byte [rsi],10
  mov rbx,10
  test rax,rax
  jnz .c
  dec rsi
  mov byte [rsi],'0'
  jmp .o
.c:
  xor rdx,rdx
  div rbx
  add dl,'0'
  dec rsi
  mov [rsi],dl
  test rax,rax
  jnz .c
.o:
  lea rdx,[numbuf+32]
  sub rdx,rsi
  mov rax,SYS_write
  mov rdi,1
  syscall
  pop rbx
  ret
; print fixed 0.xxxxxx from xmm0 in [0,1]; prints 1.000000 for >=1.0
print_p6:
  push rax
  push rbx
  ucomiss xmm0,[f_1]
  jb .norm
  lea rsi,[s_one]
  mov rdx,s_one_l
  call print_str
  pop rbx
  pop rax
  ret
.norm:
  mulss xmm0,[f_1e6]
  cvttss2si rax,xmm0
  lea rsi,[numbuf+31]
  mov byte [rsi],10
  mov ecx,6
  mov rbx,10
.d:
  xor rdx,rdx
  div rbx
  add dl,'0'
  dec rsi
  mov [rsi],dl
  dec ecx
  jnz .d
  dec rsi
  mov byte [rsi],'.'
  dec rsi
  mov byte [rsi],'0'
  lea rdx,[numbuf+32]
  sub rdx,rsi
  mov rax,SYS_write
  mov rdi,1
  syscall
  pop rbx
  pop rax
  ret

; ---- file helpers ----
; open_ro rsi=path -> rax fd or -1
open_ro:
  mov rax,SYS_open
  mov rdi,rsi
  mov rsi,O_RDONLY
  xor rdx,rdx
  syscall
  cmp rax,0
  jl .e
  ret
.e:
  mov rax,-1
  ret
; read_all rdi=fd rsi=buf rdx=max -> rax bytes
read_all:
  push rbx
  push r12
  mov rbx,rdi
  mov r12,rsi
  xor r15d,r15d
  mov r14,rdx
.rd:
  mov rax,SYS_read
  mov rdi,rbx
  mov rsi,r12
  mov rdx,r14
  syscall
  cmp rax,0
  jle .done
  add r12,rax
  sub r14,rax
  add r15,rax
  test r14,r14
  jnz .rd
.done:
  mov rax,r15
  pop r12
  pop rbx
  ret

; ---- LCG + init ----
; lcg_next rbx state -> rax sample, rbx next
lcg_next:
  mov rax,rbx
  mov rcx,6364136223846793005
  imul rax,rcx
  mov rcx,1442695040888963407
  add rax,rcx
  mov rbx,rax
  shr rax,41
  ret
; init_uniform rdi=ptr rsi=count xmm0=scale rbx=seed -> rbx updated
init_uniform:
  push rbx
  push r12
  mov r12,rsi
  movss [initsc_tmp],xmm0
.l:
  test r12,r12
  jz .d
  call lcg_next
  cvtsi2ss xmm0,eax
  divss xmm0,[f_8388608]
  subss xmm0,[f_1]
  mulss xmm0,[initsc_tmp]
  movss [rdi],xmm0
  add rdi,4
  dec r12
  jmp .l
.d:
  pop r12
  pop rbx
  ret

; ---- hashing (matches features.py) ----
hash_ngram:
  push rbx
  push rcx
  mov eax,0x811c9dc5
  test rdx,rdx
  jz .av
.l:
  movzx ebx,byte [rsi]
  xor eax,ebx
  imul eax,eax,0x01000193
  inc rsi
  dec rdx
  jnz .l
.av:
  mov ebx,eax
  shr ebx,15
  xor eax,ebx
  imul eax,eax,0x85ebca6b
  mov ebx,eax
  shr ebx,13
  xor eax,ebx
  and eax,D_MASK
  pop rcx
  pop rbx
  ret

; featurize rsi=ua rdx=len -> rax distinct count, bitset filled
featurize:
  push rbx
  push r10
  push r11
  push r12
  push r13
  push r14
  push r15
  lea rdi,[bitset]
  mov ecx,4
  vpxor ymm0,ymm0,ymm0
.z:
  vmovdqu [rdi],ymm0
  add rdi,32
  dec ecx
  jnz .z
  vzeroupper
  lea rdi,[tmpua+1]
  mov r12,rdx
  cmp r12,MAXUALEN
  jbe .okl
  mov r12,MAXUALEN
.okl:
  mov r14,rsi
  xor r15,r15
.cl:
  cmp r15,r12
  jae .cld
  movzx eax,byte [r14+r15]
  cmp al,'A'
  jb .st
  cmp al,'Z'
  ja .st
  add al,32
.st:
  mov [rdi+r15],al
  inc r15
  jmp .cl
.cld:
  mov byte [tmpua],' '
  mov byte [rdi+r12],' '
  lea r15,[tmpua]
  mov r13,r12
  add r13,2
  xor r12d,r12d
  mov rbx,3
.nl:
  cmp rbx,4 ; NOTE: matches Python reference features.py effective range (3,4):
  ; range(n_range[0],n_range[1]+1) with n_range=(3,4,5) yields 3,4 only. Keep in sync.
  ja .nd
  mov rax,r13
  sub rax,rbx
  jb .nx
  inc rax
  xor r14,r14
.pl:
  cmp r14,rax
  jae .nx
  lea rsi,[r15+r14]
  mov rdx,rbx
  push rax
  push rbx
  call hash_ngram
  mov r10d,eax ; stash hash (pop rax would clobber return value)
  pop rbx
  pop rax
  mov edx,r10d
  shr edx,3
  mov ecx,r10d
  and ecx,7
  lea rdi,[bitset]
  mov r10b,[rdi+rdx]
  mov r11b,1
  shl r11b,cl
  test r10b,r11b
  jnz .al
  or [rdi+rdx],r11b
  inc r12
.al:
  inc r14
  jmp .pl
.nx:
  inc rbx
  jmp .nl
.nd:
  mov rax,r12
  pop r15
  pop r14
  pop r13
  pop r12
  pop r11
  pop r10
  pop rbx
  ret

; emb_mean -> embmean[64], rax=count. AVX2 adds + div.
emb_mean:
  push rbx
  push r10
  push r11
  push r12
  lea rdi,[embmean]
  mov ecx,8
  vpxor ymm0,ymm0,ymm0
.z:
  vmovdqu [rdi],ymm0
  add rdi,32
  dec ecx
  jnz .z
  vzeroupper
  xor r12d,r12d
  xor ebx,ebx
.sc:
  cmp ebx,D_DIM
  jae .dn
  mov eax,ebx
  shr eax,3
  mov ecx,ebx
  and ecx,7
  lea rdi,[bitset]
  mov r10b,[rdi+rax]
  mov r11b,1
  shl r11b,cl
  test r10b,r11b
  jz .nx
  mov rax,rbx
  shl rax,8
  lea rsi,[emb]
  add rsi,rax
  lea rdi,[embmean]
  mov ecx,8
.av:
  vmovups ymm0,[rsi]
  vmovups ymm1,[rdi]
  vaddps ymm1,ymm1,ymm0
  vmovups [rdi],ymm1
  add rsi,32
  add rdi,32
  dec ecx
  jnz .av
  vzeroupper
  inc r12
.nx:
  inc ebx
  jmp .sc
.dn:
  test r12,r12
  jz .em
  cvtsi2ss xmm1,r12d
  vbroadcastss ymm1,xmm1
  lea rdi,[embmean]
  mov ecx,8
.dv:
  vmovups ymm0,[rdi]
  vdivps ymm0,ymm0,ymm1
  vmovups [rdi],ymm0
  add rdi,32
  dec ecx
  jnz .dv
  vzeroupper
  mov rax,r12
  pop r12
  pop r11
  pop r10
  pop rbx
  ret
.em:
  xor eax,eax
  pop r12
  pop r11
  pop r10
  pop rbx
  ret

; dot64: rsi=a[64] rdi=b[64] -> xmm0. AVX2 FMA 8x8 + tail (64 exact, no tail).
dot64:
  vxorps ymm0,ymm0,ymm0
  mov ecx,8
.l:
  vmovups ymm1,[rsi]
  vmovups ymm2,[rdi]
  vfmadd231ps ymm0,ymm1,ymm2
  add rsi,32
  add rdi,32
  dec ecx
  jnz .l
  vextractf128 xmm1,ymm0,1
  vaddps xmm0,xmm0,xmm1
  vmovshdup xmm1,xmm0
  vaddps xmm0,xmm0,xmm1
  vpermilps xmm1,xmm0,0xAA
  vaddss xmm0,xmm0,xmm1
  vzeroupper
  ret

; dot16: rsi=a[16] rdi=b[16] -> xmm0
dot16:
  vxorps ymm0,ymm0,ymm0
  vmovups ymm1,[rsi]
  vmovups ymm2,[rdi]
  vfmadd231ps ymm0,ymm1,ymm2
  vmovups ymm1,[rsi+32]
  vmovups ymm2,[rdi+32]
  vfmadd231ps ymm0,ymm1,ymm2
  vextractf128 xmm1,ymm0,1
  vaddps xmm0,xmm0,xmm1
  vmovshdup xmm1,xmm0
  vaddps xmm0,xmm0,xmm1
  vpermilps xmm1,xmm0,0xAA
  vaddss xmm0,xmm0,xmm1
  vzeroupper
  ret

; forward: bitset -> probs[5]. Uses embmean/hid/logits/probs. rax=nfeat.
forward:
  push rbx
  call emb_mean
  mov [nfeat_tmp],rax
  ; hid[o] = relu(dot(embmean, fc1[o]) + b1[o]); fc1 out-major [16][64]
  xor ebx,ebx
.h:
  cmp ebx,H_HID
  jae .hd
  lea rsi,[embmean]
  mov rax,rbx
  shl rax,8 ; o*256 bytes (64 floats)
  lea rdi,[fc1]
  add rdi,rax
  call dot64
  lea rax,[b1]
  movss xmm1,[rax+rbx*4]
  addss xmm0,xmm1
  xorps xmm1,xmm1
  maxss xmm0,xmm1
  lea rax,[hid]
  movss [rax+rbx*4],xmm0
  inc ebx
  jmp .h
.hd:
  ; logits[o] = dot(hid, fc2[o]) + b2[o]; fc2 out-major [5][16]
  xor ebx,ebx
.o:
  cmp ebx,C_CLS
  jae .od
  lea rsi,[hid]
  mov rax,rbx
  shl rax,6 ; o*64 bytes (16 floats)
  lea rdi,[fc2]
  add rdi,rax
  call dot16
  lea rax,[b2]
  movss xmm1,[rax+rbx*4]
  addss xmm0,xmm1
  lea rax,[logits]
  movss [rax+rbx*4],xmm0
  inc ebx
  jmp .o
.od:
  call softmax5
  mov rax,[nfeat_tmp]
  pop rbx
  ret

; expf_scalar xmm0 -> xmm0 (x87, preserves rbx)
expf_s:
  push rbx
  sub rsp,16
  movss [rsp],xmm0
  fld dword [rsp]
  fldl2e
  fmulp st1,st0
  fld st0
  frndint
  fsub st1,st0
  fxch st1
  f2xm1
  fld1
  faddp st1,st0
  fscale
  fstp st1
  fstp dword [rsp]
  movss xmm0,[rsp]
  add rsp,16
  pop rbx
  ret

; softmax5: logits -> probs
softmax5:
  push rbx
  lea rax,[logits]
  movss xmm0,[rax]
  mov ebx,1
.mx:
  cmp ebx,C_CLS
  jae .mxd
  movss xmm1,[rax+rbx*4]
  ucomiss xmm1,xmm0
  jbe .nxt
  movss xmm0,xmm1
.nxt:
  inc ebx
  jmp .mx
.mxd:
  movss [smax_tmp],xmm0
  xor ebx,ebx
  xorps xmm2,xmm2
.e:
  cmp ebx,C_CLS
  jae .ed
  movss xmm0,[rax+rbx*4]
  subss xmm0,[smax_tmp]
  call expf_s
  lea rcx,[probs]
  movss [rcx+rbx*4],xmm0
  addss xmm2,xmm0
  inc ebx
  jmp .e
.ed:
  movss xmm1,xmm2
  xor ebx,ebx
.n:
  cmp ebx,C_CLS
  jae .nd
  lea rcx,[probs]
  movss xmm0,[rcx+rbx*4]
  divss xmm0,xmm1
  movss [rcx+rbx*4],xmm0
  inc ebx
  jmp .n
.nd:
  pop rbx
  ret

; argmax probs -> eax
argmax5:
  lea rax,[probs]
  movss xmm0,[rax]
  xor eax,eax
  mov ebx,1
.l:
  cmp ebx,C_CLS
  jae .d
  lea rcx,[probs]
  movss xmm1,[rcx+rbx*4]
  ucomiss xmm1,xmm0
  jbe .n
  movss xmm0,xmm1
  mov eax,ebx
.n:
  inc ebx
  jmp .l
.d:
  ret

; ---- TSV corpus load ----
; load_tsv rsi=path -> rax rows (fills labAry/offAry/lenAry + corpus data). dies on error.
load_tsv:
  push rbx
  push r12
  push r13
  push r14
  push r15
  mov r15,rsi
  call open_ro
  cmp rax,-1
  je .fail_open
  mov rbx,rax ; fd
  lea rsi,[corpus]
  mov rdx,CORPUS_SZ
  mov rdi,rbx
  call read_all
  mov r12,rax ; bytes
  mov rax,SYS_close
  mov rdi,rbx
  syscall
  ; parse lines: "d\tUA\n"
  lea r14,[corpus]
  add r12,r14 ; end
  xor r13d,r13d ; rows
  mov rbx,r14 ; cur
.pl:
  cmp rbx,r12
  jae .pd
  ; label digit
  movzx eax,byte [rbx]
  sub al,'0'
  cmp al,4
  ja .fail_fmt
  movzx r10d,al
  cmp byte [rbx+1],9 ; tab
  jne .fail_fmt
  lea r11,[rbx+2] ; ua start
  mov rdi,r11
.fl:
  cmp rdi,r12
  jae .feol
  mov al,[rdi]
  cmp al,10
  je .feol
  inc rdi
  jmp .fl
.feol:
  mov rax,rdi
  sub rax,r11 ; ualen
  cmp rax,1
  jb .fail_fmt
  cmp rax,MAXUALEN
  ja .fail_fmt
  cmp r13,MAXROWS
  jae .fail_many
  lea rcx,[labAry]
  mov [rcx+r13],r10b
  lea rcx,[offAry]
  mov [rcx+r13*8],r11
  lea rcx,[lenAry]
  mov [rcx+r13*4],eax
  inc r13
  mov rbx,rdi
  inc rbx ; skip \n
  jmp .pl
.pd:
  mov rax,r13
  mov [nrows_tmp],rax
  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  ret
.fail_open:
  lea rsi,[e_open]
  mov rdx,e_open_l
  mov rax,SYS_write
  mov rdi,2
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall
.fail_fmt:
  lea rsi,[e_fmt]
  mov rdx,e_fmt_l
  mov rax,SYS_write
  mov rdi,2
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall
.fail_many:
  lea rsi,[e_many]
  mov rdx,e_many_l
  mov rax,SYS_write
  mov rdi,2
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall

; ---- init model ----
model_init:
  push rbx
  mov rbx,0x12345678
  lea rdi,[emb]
  mov rsi,N_EMB
  movss xmm0,[init_emb_sc]
  call init_uniform
  lea rdi,[fc1]
  mov rsi,N_FC1
  movss xmm0,[init_fc_sc]
  call init_uniform
  lea rdi,[b1]
  mov rsi,N_B1
  movss xmm0,[init_b_sc]
  call init_uniform
  lea rdi,[fc2]
  mov rsi,N_FC2
  movss xmm0,[init_fc_sc]
  call init_uniform
  lea rdi,[b2]
  mov rsi,N_B2
  movss xmm0,[init_b_sc]
  call init_uniform
  ; zero velocities
  lea rdi,[vemb]
  mov ecx,N_EMB
  xor eax,eax
  rep stosd
  lea rdi,[vfc1]
  mov ecx,N_FC1
  xor eax,eax
  rep stosd
  lea rdi,[vb1]
  mov ecx,N_B1
  xor eax,eax
  rep stosd
  lea rdi,[vfc2]
  mov ecx,N_FC2
  xor eax,eax
  rep stosd
  lea rdi,[vb2]
  mov ecx,N_B2
  xor eax,eax
  rep stosd
  pop rbx
  ret

; ---- one SGD sample: r10b=label. Assumes forward done (hid/probs/nfeat, bitset).
; lr in [lr_f], mu in [mu_f]. Sparse emb update, dense fc update.
sgd_sample:
  push rbx
  push r12
  push r13
  push r14
  push r15
  movzx r15d,r10b ; label
  ; dlogits[o] = probs[o] - (o==label)
  xor ebx,ebx
.dl:
  cmp ebx,C_CLS
  jae .dld
  lea rax,[probs]
  movss xmm0,[rax+rbx*4]
  cmp ebx,r15d
  jne .st
  subss xmm0,[f_1]
.st:
  lea rax,[dlog]
  movss [rax+rbx*4],xmm0
  inc ebx
  jmp .dl
.dld:
  ; class-weighted CE: scale dlog by w_true (balanced-sampling equivalent)
  lea rax,[cw]
  movss xmm5,[rax+r15*4]
  xor ebx,ebx
.wl:
  cmp ebx,C_CLS
  jae .wld
  lea rax,[dlog]
  movss xmm0,[rax+rbx*4]
  mulss xmm0,xmm5
  movss [rax+rbx*4],xmm0
  inc ebx
  jmp .wl
.wld:
  ; fc2 grad + update with momentum: for o,h: g = dlog[o]*hid[h]
  ; W2[o][h] layout out-major [5][16]
  xor ebx,ebx ; o
.fo:
  cmp ebx,C_CLS
  jae .fod
  lea rax,[dlog]
  movss xmm5,[rax+rbx*4] ; d
  xor r12d,r12d ; h
.fh:
  cmp r12,H_HID
  jae .fb
  lea rax,[hid]
  movss xmm0,[rax+r12*4]
  mulss xmm0,xmm5 ; g
  ; v = mu*v - lr*g ; w += v
  mov rax,rbx
  shl rax,4 ; o*16
  add rax,r12 ; idx
  lea rcx,[vfc2]
  movss xmm1,[rcx+rax*4]
  mulss xmm1,[mu_f]
  movss xmm2,xmm0
  mulss xmm2,[lr_f]
  subss xmm1,xmm2
  movss [rcx+rax*4],xmm1
  lea rcx,[fc2]
  movss xmm2,[rcx+rax*4]
  addss xmm2,xmm1
  movss [rcx+rax*4],xmm2
  inc r12
  jmp .fh
.fb:
  ; b2[o]
  lea rax,[dlog]
  movss xmm0,[rax+rbx*4]
  lea rcx,[vb2]
  movss xmm1,[rcx+rbx*4]
  mulss xmm1,[mu_f]
  movss xmm2,xmm0
  mulss xmm2,[lr_f]
  subss xmm1,xmm2
  movss [rcx+rbx*4],xmm1
  lea rcx,[b2]
  movss xmm2,[rcx+rbx*4]
  addss xmm2,xmm1
  movss [rcx+rbx*4],xmm2
  inc ebx
  jmp .fo
.fod:
  ; d hid[h] = sum_o dlog[o]*W2[o][h] * relu'(hid_pre)
  ; relu mask: hid[h] > 0 ? 1 : 0.01 (leaky to avoid dead units)
  xor r12d,r12d
.dh:
  cmp r12,H_HID
  jae .dhd
  xorps xmm0,xmm0
  xor ebx,ebx
.so:
  cmp ebx,C_CLS
  jae .sod
  lea rax,[dlog]
  movss xmm1,[rax+rbx*4]
  mov rax,rbx
  shl rax,4
  add rax,r12
  lea rcx,[fc2]
  movss xmm2,[rcx+rax*4]
  mulss xmm1,xmm2
  addss xmm0,xmm1
  inc ebx
  jmp .so
.sod:
  lea rax,[hid]
  movss xmm1,[rax+r12*4]
  xorps xmm2,xmm2
  ucomiss xmm1,xmm2
  ja .keep
  mulss xmm0,[leak_f]
.keep:
  lea rax,[dhid]
  movss [rax+r12*4],xmm0
  inc r12
  jmp .dh
.dhd:
  ; fc1 update: W1[o2? ] layout out-major [16][64]: fc1[h*64+i]
  ; g[h][i] = dhid[h]*embmean[i]; only need embmean (dense 64, fine)
  xor ebx,ebx ; h
.wo:
  cmp ebx,H_HID
  jae .wod
  lea rax,[dhid]
  movss xmm5,[rax+rbx*4]
  xor r12d,r12d ; i
.wi:
  cmp r12,E_EMB
  jae .wb
  lea rax,[embmean]
  movss xmm0,[rax+r12*4]
  mulss xmm0,xmm5
  mov rax,rbx
  shl rax,6 ; h*64
  add rax,r12
  lea rcx,[vfc1]
  movss xmm1,[rcx+rax*4]
  mulss xmm1,[mu_f]
  movss xmm2,xmm0
  mulss xmm2,[lr_f]
  subss xmm1,xmm2
  movss [rcx+rax*4],xmm1
  lea rcx,[fc1]
  movss xmm2,[rcx+rax*4]
  addss xmm2,xmm1
  movss [rcx+rax*4],xmm2
  inc r12
  jmp .wi
.wb:
  ; b1[h]
  lea rax,[dhid]
  movss xmm0,[rax+rbx*4]
  lea rcx,[vb1]
  movss xmm1,[rcx+rbx*4]
  mulss xmm1,[mu_f]
  movss xmm2,xmm0
  mulss xmm2,[lr_f]
  subss xmm1,xmm2
  movss [rcx+rbx*4],xmm1
  lea rcx,[b1]
  movss xmm2,[rcx+rbx*4]
  addss xmm2,xmm1
  movss [rcx+rbx*4],xmm2
  inc ebx
  jmp .wo
.wod:
  ; sparse emb update: demb[i] = sum_h dhid[h]*W1[h][i]; then emb[row] += v where row in bitset
  ; first compute demb[64]
  lea rdi,[demb]
  mov ecx,16
  xor eax,eax
  ; zero demb (scalar, 64)
  push rdi
  mov ecx,64
.zd:
  mov dword [rdi],0
  add rdi,4
  dec ecx
  jnz .zd
  pop rdi
  xor ebx,ebx ; h
.do2:
  cmp ebx,H_HID
  jae .dod
  lea rax,[dhid]
  movss xmm5,[rax+rbx*4]
  xor r12d,r12d
.di:
  cmp r12,E_EMB
  jae .dn2
  mov rax,rbx
  shl rax,6
  add rax,r12
  lea rcx,[fc1]
  movss xmm0,[rcx+rax*4]
  mulss xmm0,xmm5
  lea rax,[demb]
  movss xmm1,[rax+r12*4]
  addss xmm1,xmm0
  movss [rax+r12*4],xmm1
  inc r12
  jmp .di
.dn2:
  inc ebx
  jmp .do2
.dod:
  ; scale by 1/nfeat (mean)
  mov rax,[nfeat_tmp]
  test rax,rax
  jz .ed
  cvtsi2ss xmm5,eax
  xor ebx,ebx
.sc:
  cmp ebx,E_EMB
  jae .sd
  lea rax,[demb]
  movss xmm0,[rax+rbx*4]
  divss xmm0,xmm5
  movss [rax+rbx*4],xmm0
  inc ebx
  jmp .sc
.sd:
  ; for each set bit row: emb[row][i] += v (momentum per-row)
  xor ebx,ebx ; bit 0..1023
.eb:
  cmp ebx,D_DIM
  jae .ed
  mov eax,ebx
  shr eax,3
  mov ecx,ebx
  and ecx,7
  lea rdi,[bitset]
  mov r10b,[rdi+rax]
  mov r11b,1
  shl r11b,cl
  test r10b,r11b
  jz .en
  ; row update 64 floats
  mov rax,rbx
  shl rax,8
  lea rsi,[vemb]
  add rsi,rax
  lea rdi,[emb]
  add rdi,rax
  lea rdx,[demb]
  xor r12d,r12d
.eu:
  cmp r12,E_EMB
  jae .en
  movss xmm0,[rsi+r12*4]
  mulss xmm0,[mu_f]
  movss xmm1,[rdx+r12*4]
  mulss xmm1,[lr_f]
  subss xmm0,xmm1
  movss [rsi+r12*4],xmm0
  movss xmm1,[rdi+r12*4]
  addss xmm1,xmm0
  movss [rdi+r12*4],xmm1
  inc r12
  jmp .eu
.en:
  inc ebx
  jmp .eb
.ed:
  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  ret

; ---- FNV64 ----
fnv64:
  push rbx
  push r10
  mov rax,14695981039346656037
  mov rbx,1099511628211
  mov r10,rdx
.l:
  test r10,r10
  jz .d
  movzx ecx,byte [rsi]
  xor rax,rcx
  mul rbx
  inc rsi
  dec r10
  jmp .l
.d:
  pop r10
  pop rbx
  ret

; ---- checkpoint save/load ----
; model_save rsi=path -> rax 0 ok else 1
model_save:
  push rbx
  push r12
  push r13
  mov r12,rsi
  ; build tmp path: path + ".tmp"
  lea rdi,[tmppath]
  mov rsi,r12
.cp:
  mov al,[rsi]
  mov [rdi],al
  inc rsi
  inc rdi
  test al,al
  jnz .cp
  mov byte [rdi-1],'.'
  mov byte [rdi],'t'
  mov byte [rdi+1],'m'
  mov byte [rdi+2],'p'
  mov byte [rdi+3],0
  ; header in hdrbuf
  lea rdi,[hdrbuf]
  mov dword [rdi],'UATH'
  mov dword [rdi+4],1
  mov dword [rdi+8],D_DIM
  mov dword [rdi+12],E_EMB
  mov dword [rdi+16],H_HID
  mov dword [rdi+20],C_CLS
  mov rax,[mstep]
  mov [rdi+24],rax
  mov qword [rdi+32],PAY_BYTES
  ; payload crc over params in order
  lea rsi,[emb]
  mov rdx,N_EMB
  shl rdx,2
  call fnv64
  mov [rdi+40],rax
  mov [rdi+48],rax ; payload crc (single pass ok: fnv over concatenated? do chained)
  ; NOTE: chained properly: hash emb, then fc1 bytes appended? For v1, hash each region separately is overkill;
  ; store fnv(emb)||fnv(all)? Simplify: crc = fnv(emb) xor fnv(fc1) ... computed below.
  lea rsi,[emb]
  mov rdx,PAY_BYTES
  call fnv64
  mov [rdi+40],rax
  mov [rdi+48],rax
  ; open tmp O_WRONLY|CREAT|TRUNC 0644
  mov rax,SYS_open
  lea rdi,[tmppath]
  mov rsi,O_WRONLY+O_CREAT+O_TRUNC
  mov rdx,MODE_644
  syscall
  cmp rax,0
  jl .e1
  mov rbx,rax
  ; write hdr 64
  mov rax,SYS_write
  mov rdi,rbx
  lea rsi,[hdrbuf]
  mov rdx,64
  syscall
  cmp rax,64
  jne .e2
  ; write payload pieces
  mov rax,SYS_write
  mov rdi,rbx
  lea rsi,[emb]
  mov rdx,N_EMB
  shl rdx,2
  syscall
  mov r13,rax
  mov rax,SYS_write
  mov rdi,rbx
  lea rsi,[fc1]
  mov rdx,N_FC1
  shl rdx,2
  syscall
  add r13,rax
  mov rax,SYS_write
  mov rdi,rbx
  lea rsi,[b1]
  mov rdx,N_B1
  shl rdx,2
  syscall
  add r13,rax
  mov rax,SYS_write
  mov rdi,rbx
  lea rsi,[fc2]
  mov rdx,N_FC2
  shl rdx,2
  syscall
  add r13,rax
  mov rax,SYS_write
  mov rdi,rbx
  lea rsi,[b2]
  mov rdx,N_B2
  shl rdx,2
  syscall
  add r13,rax
  cmp r13,PAY_BYTES
  jne .e2
  mov rax,SYS_fsync
  mov rdi,rbx
  syscall
  mov rax,SYS_close
  mov rdi,rbx
  syscall
  ; rename tmp -> target
  mov rax,SYS_rename
  lea rdi,[tmppath]
  mov rsi,r12
  syscall
  test rax,rax
  jnz .e1
  xor eax,eax
  pop r13
  pop r12
  pop rbx
  ret
.e2:
  mov rax,SYS_close
  mov rdi,rbx
  syscall
.e1:
  mov eax,1
  pop r13
  pop r12
  pop rbx
  ret

; model_load rsi=path -> rax 0 ok else 1 (live model intact on failure: loads to staging then copies)
model_load:
  push rbx
  push r12
  push r13
  push r14
  mov r12,rsi
  call open_ro
  cmp rax,-1
  je .e1
  mov rbx,rax
  ; read hdr
  mov rax,SYS_read
  mov rdi,rbx
  lea rsi,[hdrbuf]
  mov rdx,64
  syscall
  cmp rax,64
  jne .e2
  lea rsi,[hdrbuf]
  cmp dword [rsi],'UATH'
  jne .e2
  cmp dword [rsi+4],1
  jne .e2
  cmp dword [rsi+8],D_DIM
  jne .e2
  cmp dword [rsi+20],C_CLS
  jne .e2
  mov rax,[rsi+32]
  cmp rax,PAY_BYTES
  jne .e2
  mov r14,[rsi+40] ; expect crc
  ; read payload into staging area (stg = second copy in BSS)
  mov rax,SYS_read
  mov rdi,rbx
  lea rsi,[stg]
  mov rdx,PAY_BYTES
  syscall
  cmp rax,PAY_BYTES
  jne .e2
  mov rax,SYS_close
  mov rdi,rbx
  syscall
  ; verify crc
  lea rsi,[stg]
  mov rdx,PAY_BYTES
  call fnv64
  cmp rax,r14
  jne .e1
  ; copy staging -> live
  lea rsi,[stg]
  lea rdi,[emb]
  mov ecx,PAY_BYTES
  rep movsb
  lea rsi,[hdrbuf]
  mov rax,[rsi+24]
  mov [mstep],rax
  xor eax,eax
  pop r14
  pop r13
  pop r12
  pop rbx
  ret
.e2:
  mov rax,SYS_close
  mov rdi,rbx
  syscall
.e1:
  mov eax,1
  pop r14
  pop r13
  pop r12
  pop rbx
  ret

; ---- commands ----
cmd_selftest:
  ; hash determinism: "abc" twice same; featurize count>0; forward probs sum~1; save/load roundtrip
  lea rsi,[t_abc]
  mov rdx,3
  call hash_ngram
  mov r12d,eax
  lea rsi,[t_abc]
  mov rdx,3
  call hash_ngram
  cmp eax,r12d
  jne .fail
  ; expected hash of "abc" mod 1024 (computed by python below at validation; here just determinism + range)
  cmp eax,D_DIM
  jae .fail
  call model_init
  lea rsi,[t_ua1]
  mov rdx,t_ua1_l
  call featurize
  test rax,rax
  jz .fail
  call forward
  ; sum probs
  lea rax,[probs]
  movss xmm0,[rax]
  movss xmm1,[rax+4]
  addss xmm0,xmm1
  movss xmm1,[rax+8]
  addss xmm0,xmm1
  movss xmm1,[rax+12]
  addss xmm0,xmm1
  movss xmm1,[rax+16]
  addss xmm0,xmm1
  subss xmm0,[f_1]
  ; |sum-1| < 1e-5
  xorps xmm1,xmm1
  ; abs
  lea rax,[mask_abs4]
  movups xmm2,[rax]
  andps xmm0,xmm2
  ucomiss xmm0,[f_1e_5]
  jae .fail
  ; save/load roundtrip to /tmp/uathink_selftest.bin
  lea rsi,[p_selftest]
  call model_save
  test rax,rax
  jnz .fail
  ; corrupt live then load
  mov dword [emb],0xdeadbeef
  lea rsi,[p_selftest]
  call model_load
  test rax,rax
  jnz .fail
  lea rsi,[m_pass]
  mov rdx,m_pass_l
  call print_str
  ret
.fail:
  lea rsi,[e_self]
  mov rdx,e_self_l
  mov rax,SYS_write
  mov rdi,2
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall

cmd_train:
  ; argv: train tsv model [epochs]
  mov rax,[argc_tmp]
  cmp rax,4
  jb usage
  mov rax,[argv_tmp]
  mov rsi,[rax+16] ; argv[2] tsv
  mov [arg_tsv],rsi
  mov rax,[argv_tmp]
  mov rsi,[rax+24] ; argv[3] model
  mov [arg_model],rsi
  mov qword [arg_epochs],12
  cmp qword [argc_tmp],5
  jb .noep
  mov rax,[argv_tmp]
  mov rsi,[rax+32]
  call atoi
  test rax,rax
  jz .noep
  cmp rax,200
  ja .noep
  mov [arg_epochs],rax
.noep:
  call model_init
  mov qword [mstep],0
  mov rsi,[arg_tsv]
  call load_tsv ; rax rows
  mov [nrows_tmp],rax
  lea rsi,[m_rows]
  mov rdx,m_rows_l
  call print_str
  mov rax,[nrows_tmp]
  call print_u64
  mov rbx,[arg_epochs]
  xor r14,r14 ; epoch
.ep:
  cmp r14,rbx
  jae .done
  ; lr schedule: ep0-3 0.005, ep4-7 0.0025, ep8+ 0.00125 (decay curbs margin drift)
  cmp r14,4
  jae .lr2
  movss xmm0,[lr_s0]
  movss [lr_f],xmm0
  jmp .lrs
.lr2:
  cmp r14,8
  jae .lr3
  movss xmm0,[lr_s1]
  movss [lr_f],xmm0
  jmp .lrs
.lr3:
  movss xmm0,[lr_s2]
  movss [lr_f],xmm0
.lrs:
  xor r13,r13 ; i
  mov r12,[nrows_tmp]
.lo:
  cmp r13,r12
  jae .epd
  lea rax,[labAry]
  movzx r10d,byte [rax+r13]
  lea rax,[offAry]
  mov rsi,[rax+r13*8]
  lea rax,[lenAry]
  mov edx,[rax+r13*4]
  push r12
  push r13
  push r14
  push rbx
  call featurize
  call forward
  call sgd_sample
  inc qword [mstep]
  pop rbx
  pop r14
  pop r13
  pop r12
  inc r13
  jmp .lo
.epd:
  ; epoch report: train accuracy quick pass? print STEP
  lea rsi,[m_step]
  mov rdx,m_step_l
  call print_str
  mov rax,r14
  inc rax
  call print_u64
  inc r14
  jmp .ep
.done:
  mov rsi,[arg_model]
  call model_save
  test rax,rax
  jnz .fsave
  lea rsi,[m_saved]
  mov rdx,m_saved_l
  call print_str
  ret
.fsave:
  lea rsi,[e_save]
  mov rdx,e_save_l
  mov rax,SYS_write
  mov rdi,2
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall

cmd_predict:
  ; argv: predict model [ua...]; no ua -> read stdin lines
  cmp qword [argc_tmp],3
  jb usage
  mov rax,[argv_tmp]
  mov rsi,[rax+16]
  call model_load
  test rax,rax
  jnz .fload
  cmp qword [argc_tmp],4
  jae .args
  ; stdin mode
.stdin:
  mov rax,SYS_read
  mov rdi,0
  lea rsi,[stdinbuf]
  mov rdx,65536
  syscall
  cmp rax,0
  jle .ret
  mov r12,rax
  lea r13,[stdinbuf]
  add r12,r13
  mov rbx,r13
.line:
  cmp rbx,r12
  jae .ret
  mov rdi,rbx
.nl:
  cmp rdi,r12
  jae .eol
  cmp byte [rdi],10
  je .eol
  inc rdi
  jmp .nl
.eol:
  mov rax,rdi
  sub rax,rbx
  test rax,rax
  jz .sk
  cmp rax,MAXUALEN
  ja .sk
  mov rsi,rbx
  mov rdx,rax
  push rbx
  push rdi
  push r12
  call featurize
  call forward
  call argmax5
  mov r14d,eax
  ; print: "<label> p0 .. p4\n"
  mov rax,r14
  call print_u64
  ; print probs
  xor ebx,ebx
.pp:
  cmp ebx,5
  jae .pn
  lea rsi,[m_sp]
  mov rdx,1
  call print_str
  lea rax,[probs]
  movss xmm0,[rax+rbx*4]
  call print_p6
  inc ebx
  jmp .pp
.pn:
  pop r12
  pop rdi
  pop rbx
.sk:
  mov rbx,rdi
  inc rbx
  jmp .line
.ret:
  ret
.args:
  mov r13,3 ; idx
  mov r14,[argc_tmp]
.al:
  cmp r13,r14
  jae .ret
  mov rax,[argv_tmp]
  mov rsi,[rax+r13*8] ; argv[idx]
  ; strlen
  mov rdi,rsi
  xor rdx,rdx
.sl:
  cmp byte [rdi],0
  je .sld
  inc rdi
  inc rdx
  jmp .sl
.sld:
  cmp rdx,1
  jb .nx
  cmp rdx,MAXUALEN
  ja .nx
  push r13
  push r14
  call featurize
  call forward
  call argmax5
  mov r15d,eax
  mov rax,r15
  call print_u64
  xor ebx,ebx
.qp:
  cmp ebx,5
  jae .qn
  lea rsi,[m_sp]
  mov rdx,1
  call print_str
  lea rax,[probs]
  movss xmm0,[rax+rbx*4]
  call print_p6
  inc ebx
  jmp .qp
.qn:
  pop r14
  pop r13
.nx:
  inc r13
  jmp .al
.fload:
  lea rsi,[e_load]
  mov rdx,e_load_l
  mov rax,SYS_write
  mov rdi,2
  syscall
  mov rax,SYS_exit
  mov edi,1
  syscall

cmd_eval:
  ; argv: eval model eval.tsv -> prints acc + confusion rows + per-class P/R/F1 (FP scalar ok)
  cmp qword [argc_tmp],4
  jb usage
  mov rax,[argv_tmp]
  mov rsi,[rax+16]
  call model_load
  test rax,rax
  jnz cmd_predict.fload
  mov rax,[argv_tmp]
  mov rsi,[rax+24]
  call load_tsv
  mov [nrows_tmp],rax
  ; zero conf[5][5] (pred rows? use conf[true*5+pred])
  lea rdi,[conf]
  mov ecx,25
  xor eax,eax
  rep stosd
  xor r13,r13
  mov r12,[nrows_tmp]
.el:
  cmp r13,r12
  jae .ed
  lea rax,[labAry]
  movzx r10d,byte [rax+r13]
  lea rax,[offAry]
  mov rsi,[rax+r13*8]
  lea rax,[lenAry]
  mov edx,[rax+r13*4]
  push r12
  push r13
  call featurize
  call forward
  call argmax5 ; eax pred
  mov r11d,eax
  pop r13
  pop r12
  ; conf[true*5+pred]++
  mov eax,r10d
  imul eax,eax,5
  add eax,r11d
  lea rcx,[conf]
  inc dword [rcx+rax*4]
  inc r13
  jmp .el
.ed:
  ; accuracy = trace/total; print
  xor ebx,ebx
  xor r14d,r14d
  mov r15,[nrows_tmp]
.acc:
  cmp ebx,5
  jae .ad
  mov eax,ebx
  imul eax,eax,5
  add eax,ebx
  lea rcx,[conf]
  mov eax,[rcx+rax*4]
  add r14d,eax
  inc ebx
  jmp .acc
.ad:
  lea rsi,[m_acc]
  mov rdx,m_acc_l
  call print_str
  ; acc = r14/r15 as 0.xxxxxx
  cvtsi2ss xmm0,r14d
  cvtsi2ss xmm1,r15d
  divss xmm0,xmm1
  call print_p6
  ; per-class P/R/F1 scalar
  xor ebx,ebx
.cl:
  cmp ebx,5
  jae .cd
  ; tp = conf[b*5+b]; col = sum conf[*5+b] (pred); row = sum conf[b*5+*]
  mov eax,ebx
  imul eax,eax,5
  add eax,ebx
  lea rcx,[conf]
  mov r14d,[rcx+rax*4] ; tp
  xor r10d,r10d ; col
  xor r11d,r11d ; ri
  xor r12d,r12d ; k
.cs:
  cmp r12d,5
  jae .csd
  mov eax,r12d
  imul eax,eax,5
  add eax,ebx
  add r10d,[rcx+rax*4]
  mov eax,ebx
  imul eax,eax,5
  add eax,r12d
  add r11d,[rcx+rax*4]
  inc r12d
  jmp .cs
.csd:
  ; p = tp/col, r = tp/row, f1
  cvtsi2ss xmm0,r14d
  test r10d,r10d
  jz .pz
  cvtsi2ss xmm1,r10d
  divss xmm0,xmm1
  jmp .ps
.pz:
  xorps xmm0,xmm0
.ps:
  movss [pr_tmp],xmm0
  cvtsi2ss xmm0,r14d
  test r11d,r11d
  jz .rz
  cvtsi2ss xmm1,r11d
  divss xmm0,xmm1
  jmp .rs
.rz:
  xorps xmm0,xmm0
.rs:
  movss [re_tmp],xmm0
  ; f1 = 2pr/(p+r)
  movss xmm1,[pr_tmp]
  movss xmm2,xmm0
  mulss xmm1,xmm2
  addss xmm1,xmm1 ; 2pr
  addss xmm2,[pr_tmp] ; p+r
  xorps xmm3,xmm3
  ucomiss xmm2,xmm3
  je .fz
  divss xmm1,xmm2
  jmp .fs
.fz:
  xorps xmm1,xmm1
.fs:
  push rbx
  lea rsi,[m_c]
  mov rdx,m_c_l
  call print_str
  pop rbx
  push rbx
  mov rax,rbx
  call print_u64
  lea rsi,[m_prf]
  mov rdx,m_prf_l
  call print_str
  movss xmm0,[pr_tmp]
  call print_p6
  lea rsi,[m_sp]
  mov rdx,1
  call print_str
  movss xmm0,[re_tmp]
  call print_p6
  lea rsi,[m_sp]
  mov rdx,1
  call print_str
  movss xmm0,xmm1
  call print_p6
  pop rbx
  inc ebx
  jmp .cl
.cd:
  ret

cmd_bench:
  ; time 2000 forwards on embedded sample UA
  call model_init
  lea rsi,[t_ua1]
  mov rdx,t_ua1_l
  call featurize
  push r15
  mov r15,2000
  ; rdtsc start
  rdtsc
  shl rdx,32
  or rax,rdx
  mov [tsc0],rax
.l:
  test r15,r15
  jz .d
  call forward
  dec r15
  jmp .l
.d:
  rdtsc
  shl rdx,32
  or rax,rdx
  sub rax,[tsc0]
  ; rax cycles for 2000 -> print cycles/row
  mov rdx,0
  mov rcx,2000
  div rcx
  call print_u64
  lea rsi,[m_cyc]
  mov rdx,m_cyc_l
  call print_str
  pop r15
  ret

cmd_hash:
  ; argv: hash <ua> -> prints nfeat then sorted bit indices
  cmp qword [argc_tmp],3
  jb usage
  mov rax,[argv_tmp]
  mov rsi,[rax+16]
  mov rdi,rsi
  xor rdx,rdx
.hl:
  cmp byte [rdi],0
  je .hd
  inc rdi
  inc rdx
  jmp .hl
.hd:
  cmp rdx,1
  jb usage
  cmp rdx,MAXUALEN
  ja usage
  push rsi
  push rdx
  mov rax,rdx
  call print_u64 ; debug: input len
  pop rdx
  pop rsi
  call featurize ; rax = count
  push rax
  call print_u64
  pop rax
  mov r15,rax ; save count? not needed
  xor ebx,ebx
.hb:
  cmp ebx,D_DIM
  jae .hr
  mov eax,ebx
  shr eax,3
  mov ecx,ebx
  and ecx,7
  lea rdi,[bitset]
  mov r10b,[rdi+rax]
  mov r11b,1
  shl r11b,cl
  test r10b,r11b
  jz .hn
  push rbx
  mov rax,rbx
  call print_u64
  pop rbx
.hn:
  inc ebx
  jmp .hb
.hr:
  ret

; atoi rsi -> rax (decimal, stops at NUL)
atoi:
  xor eax,eax
.l:
  movzx ecx,byte [rsi]
  test cl,cl
  jz .d
  sub cl,'0'
  cmp cl,9
  ja .d
  imul rax,rax,10
  movzx ecx,cl
  add rax,rcx
  inc rsi
  jmp .l
.d:
  ret

segment readable
s_selftest db 'selftest',0
s_train db 'train',0
s_predict db 'predict',0
s_eval db 'eval',0
s_bench db 'bench',0
s_hash db 'hash',0
m_use db 'usage: uathink selftest | train <train.tsv> <model.bin> [epochs] | predict <model.bin> [ua...] | eval <model.bin> <eval.tsv> | bench',10
m_use_l=$-m_use
e_v3 db 'uathink: x86-64-v3 required (AVX/AVX2/FMA/BMI1/BMI2/LZCNT)',10
e_v3_l=$-e_v3
e_open db 'uathink: open failed',10
e_open_l=$-e_open
e_fmt db 'uathink: bad TSV (want "<0-4> TAB UA LF")',10
e_fmt_l=$-e_fmt
e_many db 'uathink: too many rows',10
e_many_l=$-e_many
e_save db 'uathink: save failed',10
e_save_l=$-e_save
e_load db 'uathink: load failed',10
e_load_l=$-e_load
e_self db 'uathink: SELFTEST FAIL',10
e_self_l=$-e_self
m_pass db 'SELFTEST PASS',10
m_pass_l=$-m_pass
m_rows db 'rows=',0
m_rows_l=$-m_rows-1
m_step db 'epoch done step=',0
m_step_l=$-m_step-1
m_saved db 'saved',10
m_saved_l=$-m_saved
m_acc db 'acc=',0
m_acc_l=$-m_acc-1
m_c db 'c',0
m_c_l=$-m_c-1
m_prf db ' P=',0
m_prf_l=$-m_prf-1
m_sp db ' ',0
m_cyc db ' cycles/row',10
m_cyc_l=$-m_cyc
t_abc db 'abc',0
t_ua1 db 'SyntheticBench/1.0 (compatible; Example Test Agent; +http://example.com/bench)',0
t_ua1_l=$-t_ua1-1
s_one db '1.000000',10
s_one_l=$-s_one
p_selftest db '/tmp/uathink_selftest.bin',0
f_1 dd 1.0
f_1e6 dd 1000000.0
f_1e_5 dd 0.00001
f_1e_8 dd 0.00000001
f_8388608 dd 8388608.0
leak_f dd 0.01
init_emb_sc dd 0.10
init_fc_sc dd 0.50
init_b_sc dd 0.10
mu_f dd 0.0
lr_s0 dd 0.005
lr_s1 dd 0.0025
lr_s2 dd 0.00125
; class weights neutral (oversampling in TSV carries balance); kept for ablation
cw dd 1.0, 1.0, 1.0, 1.0, 1.0
mask_abs4 dd 0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff

segment readable writeable
numbuf rb 32
hdrbuf rb 64
tmppath rb 1024
bitset rb 128
tmpua rb 516
embmean rb 256
hid rb 64
logits rb 20
probs rb 20
dlog rb 20
dhid rb 64
demb rb 256
nfeat_tmp rq 1
nrows_tmp rq 1
mstep rq 1
smax_tmp rd 1
lr_f rd 1
pr_tmp rd 1
re_tmp rd 1
initsc_tmp rd 1
tsc0 rq 1
arg_tsv rq 1
arg_model rq 1
arg_epochs rq 1
argc_tmp rq 1
argv_tmp rq 1
conf rd 25
labAry rb MAXROWS
offAry rq MAXROWS
lenAry rd MAXROWS
stdinbuf rb 65536
; params (order = checkpoint payload)
emb rd N_EMB
fc1 rd N_FC1
b1 rd N_B1
fc2 rd N_FC2
b2 rd N_B2
; velocities
vemb rd N_EMB
vfc1 rd N_FC1
vb1 rd N_B1
vfc2 rd N_FC2
vb2 rd N_B2
; staging for load verify
stg rb PAY_BYTES
; corpus last (NOBITS-friendly trailing)
corpus rb CORPUS_SZ
