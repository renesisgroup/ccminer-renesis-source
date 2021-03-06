/**
 * x97 SONO
 **/

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"

#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"

#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#define NBN 2

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void x13_hamsi_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x17_haval256_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* resNonce, uint64_t target);



extern "C" void sonoa_hash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;
	sph_haval256_5_context ctx_haval;



	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*)hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash); 

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);

	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512, (const void*)hash, 64);
	sph_sha512_close(&ctx_sha512, (void*)hash);

	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);

	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);

	sph_sha512(&ctx_sha512, (const void*)hash, 64);
	sph_sha512_close(&ctx_sha512, (void*)hash);
	
	sph_haval256_5_init(&ctx_haval);
	sph_haval256_5(&ctx_haval, (const void*)hash, 64);
	sph_haval256_5_close(&ctx_haval, (void*)hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_sonoa(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	uint32_t default_throughput;
	if (device_sm[dev_id] <= 500) default_throughput = 1 << 18;
	else if (device_sm[dev_id] <= 520) default_throughput = 1 << 18;
	else if (device_sm[dev_id]>520) default_throughput = (1 << 19) + (1 << 18);

	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	throughput &= 0xFFFFFF70;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		x11_simd_echo_512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		sleep(2);
		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));

	do {
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);		
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);	
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);		
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_skein512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x17_haval256_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id],d_resNonce[thr_id],*(uint64_t*)&ptarget[6]);

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			sonoa_hash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput + 1;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					pdata[21] = startNounce+h_resNonce[thr_id][1];
					be32enc(&endiandata[19], pdata[21]);
					sonoa_hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19],pdata[21]);
					}
					res++;
				}
				return res;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));

	*hashes_done = pdata[19] - first_nonce + 1;

	return 0;
}

extern "C" void free_sonoa(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_hash[thr_id]);

	x11_simd_echo_512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);
	cudaDeviceSynchronize();
	init[thr_id] = false;
}
