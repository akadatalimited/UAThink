FASM ?= fasm
FASM_MEM ?= 131072

all: uathink

uathink: uathink.asm
	$(FASM) -m $(FASM_MEM) uathink.asm uathink
	chmod +x uathink

test: uathink
	./uathink selftest
	sh tests/test.sh

bench: uathink
	./uathink bench

clean:
	rm -f uathink /tmp/uathink_*.bin /tmp/uathink-release-test*.bin

.PHONY: all test bench clean
