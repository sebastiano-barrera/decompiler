#include <stdint.h>

// [limitation--no-relocatable] due to a known limitation, we can't process
// relocatable executables (we can't run relocations at all).
// adding main() allows us to compile this to a 'full' executable rather than a .o
int main() {}


void func000(uint8_t arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0)) :  /* clobbers */ "r10");
}

void func001(uint16_t arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0)) :  /* clobbers */ "r10");
}

void func002(uint32_t arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0)) :  /* clobbers */ "r10");
}

void func003(uint64_t arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0)) :  /* clobbers */ "r10");
}

void func004(void* arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0)) :  /* clobbers */ "r10");
}

void func005(float arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0)) :  /* clobbers */ "r10");
}

void func006(double arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0)) :  /* clobbers */ "r10");
}
struct small {
  void* member0;
  float member1;
  uint8_t member2;
};
void func008(struct small arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member0)) :  /* clobbers */ "r10");
}
void func009(struct small arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0.member1)) :  /* clobbers */ "r10");
}
void func010(struct small arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member2)) :  /* clobbers */ "r10");
}
struct small_xmms {
  float member0;
  double member1;
};
void func012(struct small_xmms arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0.member0)) :  /* clobbers */ "r10");
}
void func013(struct small_xmms arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0.member1)) :  /* clobbers */ "r10");
}
struct big {
  float member0;
  double member1;
  void* member2;
  uint8_t member3;
  uint8_t member4[3];
};
void func015(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0.member0)) :  /* clobbers */ "r10");
}
void func016(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg0.member1)) :  /* clobbers */ "r10");
}
void func017(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member2)) :  /* clobbers */ "r10");
}
void func018(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member3)) :  /* clobbers */ "r10");
}
void func019(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member4[0])) :  /* clobbers */ "r10");
}
void func020(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member4[1])) :  /* clobbers */ "r10");
}
void func021(struct big arg0) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg0.member4[2])) :  /* clobbers */ "r10");
}
void func022(struct small arg0, uint8_t arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1)) :  /* clobbers */ "r10");
}
void func023(struct small arg0, uint16_t arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1)) :  /* clobbers */ "r10");
}
void func024(struct small arg0, uint32_t arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1)) :  /* clobbers */ "r10");
}
void func025(struct small arg0, uint64_t arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1)) :  /* clobbers */ "r10");
}
void func026(struct small arg0, void* arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1)) :  /* clobbers */ "r10");
}
void func027(struct small arg0, float arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1)) :  /* clobbers */ "r10");
}
void func028(struct small arg0, double arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1)) :  /* clobbers */ "r10");
}
void func030(struct small arg0, struct small arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member0)) :  /* clobbers */ "r10");
}
void func031(struct small arg0, struct small arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1.member1)) :  /* clobbers */ "r10");
}
void func032(struct small arg0, struct small arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member2)) :  /* clobbers */ "r10");
}
void func034(struct small arg0, struct small_xmms arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1.member0)) :  /* clobbers */ "r10");
}
void func035(struct small arg0, struct small_xmms arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1.member1)) :  /* clobbers */ "r10");
}
void func037(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1.member0)) :  /* clobbers */ "r10");
}
void func038(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg1.member1)) :  /* clobbers */ "r10");
}
void func039(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member2)) :  /* clobbers */ "r10");
}
void func040(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member3)) :  /* clobbers */ "r10");
}
void func041(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member4[0])) :  /* clobbers */ "r10");
}
void func042(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member4[1])) :  /* clobbers */ "r10");
}
void func043(struct small arg0, struct big arg1) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg1.member4[2])) :  /* clobbers */ "r10");
}
void func044(void* arg0, struct small arg1, uint8_t arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2)) :  /* clobbers */ "r10");
}
void func045(void* arg0, struct small arg1, uint16_t arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2)) :  /* clobbers */ "r10");
}
void func046(void* arg0, struct small arg1, uint32_t arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2)) :  /* clobbers */ "r10");
}
void func047(void* arg0, struct small arg1, uint64_t arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2)) :  /* clobbers */ "r10");
}
void func048(void* arg0, struct small arg1, void* arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2)) :  /* clobbers */ "r10");
}
void func049(void* arg0, struct small arg1, float arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2)) :  /* clobbers */ "r10");
}
void func050(void* arg0, struct small arg1, double arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2)) :  /* clobbers */ "r10");
}
void func052(void* arg0, struct small arg1, struct small arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member0)) :  /* clobbers */ "r10");
}
void func053(void* arg0, struct small arg1, struct small arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2.member1)) :  /* clobbers */ "r10");
}
void func054(void* arg0, struct small arg1, struct small arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member2)) :  /* clobbers */ "r10");
}
void func056(void* arg0, struct small arg1, struct small_xmms arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2.member0)) :  /* clobbers */ "r10");
}
void func057(void* arg0, struct small arg1, struct small_xmms arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2.member1)) :  /* clobbers */ "r10");
}
void func059(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2.member0)) :  /* clobbers */ "r10");
}
void func060(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg2.member1)) :  /* clobbers */ "r10");
}
void func061(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member2)) :  /* clobbers */ "r10");
}
void func062(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member3)) :  /* clobbers */ "r10");
}
void func063(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member4[0])) :  /* clobbers */ "r10");
}
void func064(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member4[1])) :  /* clobbers */ "r10");
}
void func065(void* arg0, struct small arg1, struct big arg2) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg2.member4[2])) :  /* clobbers */ "r10");
}
void func066(void* arg0, void* arg1, struct small arg2, uint8_t arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3)) :  /* clobbers */ "r10");
}
void func067(void* arg0, void* arg1, struct small arg2, uint16_t arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3)) :  /* clobbers */ "r10");
}
void func068(void* arg0, void* arg1, struct small arg2, uint32_t arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3)) :  /* clobbers */ "r10");
}
void func069(void* arg0, void* arg1, struct small arg2, uint64_t arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3)) :  /* clobbers */ "r10");
}
void func070(void* arg0, void* arg1, struct small arg2, void* arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3)) :  /* clobbers */ "r10");
}
void func071(void* arg0, void* arg1, struct small arg2, float arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3)) :  /* clobbers */ "r10");
}
void func072(void* arg0, void* arg1, struct small arg2, double arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3)) :  /* clobbers */ "r10");
}
void func074(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member0)) :  /* clobbers */ "r10");
}
void func075(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3.member1)) :  /* clobbers */ "r10");
}
void func076(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member2)) :  /* clobbers */ "r10");
}
void func078(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3.member0)) :  /* clobbers */ "r10");
}
void func079(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3.member1)) :  /* clobbers */ "r10");
}
void func081(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3.member0)) :  /* clobbers */ "r10");
}
void func082(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg3.member1)) :  /* clobbers */ "r10");
}
void func083(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member2)) :  /* clobbers */ "r10");
}
void func084(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member3)) :  /* clobbers */ "r10");
}
void func085(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member4[0])) :  /* clobbers */ "r10");
}
void func086(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member4[1])) :  /* clobbers */ "r10");
}
void func087(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg3.member4[2])) :  /* clobbers */ "r10");
}
void func088(void* arg0, void* arg1, void* arg2, struct small arg3, uint8_t arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4)) :  /* clobbers */ "r10");
}
void func089(void* arg0, void* arg1, void* arg2, struct small arg3, uint16_t arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4)) :  /* clobbers */ "r10");
}
void func090(void* arg0, void* arg1, void* arg2, struct small arg3, uint32_t arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4)) :  /* clobbers */ "r10");
}
void func091(void* arg0, void* arg1, void* arg2, struct small arg3, uint64_t arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4)) :  /* clobbers */ "r10");
}
void func092(void* arg0, void* arg1, void* arg2, struct small arg3, void* arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4)) :  /* clobbers */ "r10");
}
void func093(void* arg0, void* arg1, void* arg2, struct small arg3, float arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4)) :  /* clobbers */ "r10");
}
void func094(void* arg0, void* arg1, void* arg2, struct small arg3, double arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4)) :  /* clobbers */ "r10");
}
void func096(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member0)) :  /* clobbers */ "r10");
}
void func097(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4.member1)) :  /* clobbers */ "r10");
}
void func098(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member2)) :  /* clobbers */ "r10");
}
void func100(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4.member0)) :  /* clobbers */ "r10");
}
void func101(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4.member1)) :  /* clobbers */ "r10");
}
void func103(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4.member0)) :  /* clobbers */ "r10");
}
void func104(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg4.member1)) :  /* clobbers */ "r10");
}
void func105(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member2)) :  /* clobbers */ "r10");
}
void func106(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member3)) :  /* clobbers */ "r10");
}
void func107(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member4[0])) :  /* clobbers */ "r10");
}
void func108(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member4[1])) :  /* clobbers */ "r10");
}
void func109(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg4.member4[2])) :  /* clobbers */ "r10");
}
void func110(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, uint8_t arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5)) :  /* clobbers */ "r10");
}
void func111(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, uint16_t arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5)) :  /* clobbers */ "r10");
}
void func112(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, uint32_t arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5)) :  /* clobbers */ "r10");
}
void func113(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, uint64_t arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5)) :  /* clobbers */ "r10");
}
void func114(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, void* arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5)) :  /* clobbers */ "r10");
}
void func115(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, float arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5)) :  /* clobbers */ "r10");
}
void func116(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, double arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5)) :  /* clobbers */ "r10");
}
void func118(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member0)) :  /* clobbers */ "r10");
}
void func119(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5.member1)) :  /* clobbers */ "r10");
}
void func120(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member2)) :  /* clobbers */ "r10");
}
void func122(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5.member0)) :  /* clobbers */ "r10");
}
void func123(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5.member1)) :  /* clobbers */ "r10");
}
void func125(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5.member0)) :  /* clobbers */ "r10");
}
void func126(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg5.member1)) :  /* clobbers */ "r10");
}
void func127(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member2)) :  /* clobbers */ "r10");
}
void func128(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member3)) :  /* clobbers */ "r10");
}
void func129(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member4[0])) :  /* clobbers */ "r10");
}
void func130(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member4[1])) :  /* clobbers */ "r10");
}
void func131(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg5.member4[2])) :  /* clobbers */ "r10");
}
void func132(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, uint8_t arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6)) :  /* clobbers */ "r10");
}
void func133(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, uint16_t arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6)) :  /* clobbers */ "r10");
}
void func134(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, uint32_t arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6)) :  /* clobbers */ "r10");
}
void func135(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, uint64_t arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6)) :  /* clobbers */ "r10");
}
void func136(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, void* arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6)) :  /* clobbers */ "r10");
}
void func137(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, float arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6)) :  /* clobbers */ "r10");
}
void func138(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, double arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6)) :  /* clobbers */ "r10");
}
void func140(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member0)) :  /* clobbers */ "r10");
}
void func141(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6.member1)) :  /* clobbers */ "r10");
}
void func142(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member2)) :  /* clobbers */ "r10");
}
void func144(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6.member0)) :  /* clobbers */ "r10");
}
void func145(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6.member1)) :  /* clobbers */ "r10");
}
void func147(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6.member0)) :  /* clobbers */ "r10");
}
void func148(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((double)(arg6.member1)) :  /* clobbers */ "r10");
}
void func149(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member2)) :  /* clobbers */ "r10");
}
void func150(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member3)) :  /* clobbers */ "r10");
}
void func151(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member4[0])) :  /* clobbers */ "r10");
}
void func152(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member4[1])) :  /* clobbers */ "r10");
}
void func153(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  asm("mov r10, %0" : /* outputs */ : /* inputs */ "irm" ((uint64_t)(arg6.member4[2])) :  /* clobbers */ "r10");
}
