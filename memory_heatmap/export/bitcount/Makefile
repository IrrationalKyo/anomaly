FILE = bitcnt_1.c bitcnt_2.c bitcnt_3.c bitcnt_4.c bitcnts.c bitfiles.c bitstrng.c bstr_i.c 
ACC = arm-linux-gnueabi-gcc
CXX = gcc

bitcnts: ${FILE} Makefile
	 ${CXX} -static ${FILE} -O3 -o bitcnts

clean:
	rm -rf bitcnts output*
