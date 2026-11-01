// Multi-threaded CPU read bandwidth: each thread sums its slice of a buffer.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
static uint8_t *buf; static size_t total; static int nt;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *work(void *arg){
  size_t id=(size_t)arg, n=total/nt; const uint64x2_t *p=(const uint64x2_t*)(buf+id*n);
  uint64x2_t a=vdupq_n_u64(0),b=a,c=a,d=a;
  for(size_t i=0;i<n/16;i+=4){a=veorq_u64(a,p[i]);b=veorq_u64(b,p[i+1]);c=veorq_u64(c,p[i+2]);d=veorq_u64(d,p[i+3]);}
  uint64x2_t r=veorq_u64(veorq_u64(a,b),veorq_u64(c,d));
  return (void*)(uintptr_t)(vgetq_lane_u64(r,0)^vgetq_lane_u64(r,1));
}
int main(int argc,char**argv){
  total=(size_t)1<<30; buf=aligned_alloc(4096,total); memset(buf,1,total);
  int counts[]={1,2,4,6,8,10};
  for(int k=0;k<6;k++){nt=counts[k]; double best=1e9;
    for(int r=0;r<5;r++){pthread_t th[16]; double t=now();
      for(int i=0;i<nt;i++)pthread_create(&th[i],0,work,(void*)(size_t)i);
      for(int i=0;i<nt;i++)pthread_join(th[i],0);
      t=now()-t; if(t<best)best=t;}
    printf("threads=%2d read %6.1f GB/s\n",nt,total/best/1e9);}
}
