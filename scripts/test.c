#include <stdint.h>

char func0(char arg0) {
  return arg0;
}

short int func1(short int arg0) {
  return arg0;
}

int func2(int arg0) {
  return arg0;
}

long long int func3(long long int arg0) {
  return arg0;
}

void* func4(void* arg0) {
  return arg0;
}

float func5(float arg0) {
  return arg0;
}

double func6(double arg0) {
  return arg0;
}
struct small {
  void* member0;
  float member1;
  uint8_t member2;
};
void* func7(struct small arg0) {
  return arg0.member0;
}
float func8(struct small arg0) {
  return arg0.member1;
}
uint8_t func9(struct small arg0) {
  return arg0.member2;
}
struct small_xmms {
  float member0;
  double member1;
};
float func10(struct small_xmms arg0) {
  return arg0.member0;
}
double func11(struct small_xmms arg0) {
  return arg0.member1;
}
struct big {
  float member0;
  double member1;
  void* member2;
  uint8_t member3;
  uint8_t member4[3];
};
float func12(struct big arg0) {
  return arg0.member0;
}
double func13(struct big arg0) {
  return arg0.member1;
}
void* func14(struct big arg0) {
  return arg0.member2;
}
uint8_t func15(struct big arg0) {
  return arg0.member3;
}
uint8_t func16(struct big arg0) {
  return arg0.member4[0];
}
uint8_t func17(struct big arg0) {
  return arg0.member4[1];
}
uint8_t func18(struct big arg0) {
  return arg0.member4[2];
}
char func19(struct small arg0, char arg1) {
  return arg1;
}
short int func20(struct small arg0, short int arg1) {
  return arg1;
}
int func21(struct small arg0, int arg1) {
  return arg1;
}
long long int func22(struct small arg0, long long int arg1) {
  return arg1;
}
void* func23(struct small arg0, void* arg1) {
  return arg1;
}
float func24(struct small arg0, float arg1) {
  return arg1;
}
double func25(struct small arg0, double arg1) {
  return arg1;
}
void* func26(struct small arg0, struct small arg1) {
  return arg1.member0;
}
float func27(struct small arg0, struct small arg1) {
  return arg1.member1;
}
uint8_t func28(struct small arg0, struct small arg1) {
  return arg1.member2;
}
float func29(struct small arg0, struct small_xmms arg1) {
  return arg1.member0;
}
double func30(struct small arg0, struct small_xmms arg1) {
  return arg1.member1;
}
float func31(struct small arg0, struct big arg1) {
  return arg1.member0;
}
double func32(struct small arg0, struct big arg1) {
  return arg1.member1;
}
void* func33(struct small arg0, struct big arg1) {
  return arg1.member2;
}
uint8_t func34(struct small arg0, struct big arg1) {
  return arg1.member3;
}
uint8_t func35(struct small arg0, struct big arg1) {
  return arg1.member4[0];
}
uint8_t func36(struct small arg0, struct big arg1) {
  return arg1.member4[1];
}
uint8_t func37(struct small arg0, struct big arg1) {
  return arg1.member4[2];
}
char func38(void* arg0, struct small arg1, char arg2) {
  return arg2;
}
short int func39(void* arg0, struct small arg1, short int arg2) {
  return arg2;
}
int func40(void* arg0, struct small arg1, int arg2) {
  return arg2;
}
long long int func41(void* arg0, struct small arg1, long long int arg2) {
  return arg2;
}
void* func42(void* arg0, struct small arg1, void* arg2) {
  return arg2;
}
float func43(void* arg0, struct small arg1, float arg2) {
  return arg2;
}
double func44(void* arg0, struct small arg1, double arg2) {
  return arg2;
}
void* func45(void* arg0, struct small arg1, struct small arg2) {
  return arg2.member0;
}
float func46(void* arg0, struct small arg1, struct small arg2) {
  return arg2.member1;
}
uint8_t func47(void* arg0, struct small arg1, struct small arg2) {
  return arg2.member2;
}
float func48(void* arg0, struct small arg1, struct small_xmms arg2) {
  return arg2.member0;
}
double func49(void* arg0, struct small arg1, struct small_xmms arg2) {
  return arg2.member1;
}
float func50(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member0;
}
double func51(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member1;
}
void* func52(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member2;
}
uint8_t func53(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member3;
}
uint8_t func54(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member4[0];
}
uint8_t func55(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member4[1];
}
uint8_t func56(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member4[2];
}
char func57(void* arg0, void* arg1, struct small arg2, char arg3) {
  return arg3;
}
short int func58(void* arg0, void* arg1, struct small arg2, short int arg3) {
  return arg3;
}
int func59(void* arg0, void* arg1, struct small arg2, int arg3) {
  return arg3;
}
long long int func60(void* arg0, void* arg1, struct small arg2, long long int arg3) {
  return arg3;
}
void* func61(void* arg0, void* arg1, struct small arg2, void* arg3) {
  return arg3;
}
float func62(void* arg0, void* arg1, struct small arg2, float arg3) {
  return arg3;
}
double func63(void* arg0, void* arg1, struct small arg2, double arg3) {
  return arg3;
}
void* func64(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3.member0;
}
float func65(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3.member1;
}
uint8_t func66(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3.member2;
}
float func67(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  return arg3.member0;
}
double func68(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  return arg3.member1;
}
float func69(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member0;
}
double func70(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member1;
}
void* func71(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member2;
}
uint8_t func72(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member3;
}
uint8_t func73(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member4[0];
}
uint8_t func74(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member4[1];
}
uint8_t func75(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member4[2];
}
char func76(void* arg0, void* arg1, void* arg2, struct small arg3, char arg4) {
  return arg4;
}
short int func77(void* arg0, void* arg1, void* arg2, struct small arg3, short int arg4) {
  return arg4;
}
int func78(void* arg0, void* arg1, void* arg2, struct small arg3, int arg4) {
  return arg4;
}
long long int func79(void* arg0, void* arg1, void* arg2, struct small arg3, long long int arg4) {
  return arg4;
}
void* func80(void* arg0, void* arg1, void* arg2, struct small arg3, void* arg4) {
  return arg4;
}
float func81(void* arg0, void* arg1, void* arg2, struct small arg3, float arg4) {
  return arg4;
}
double func82(void* arg0, void* arg1, void* arg2, struct small arg3, double arg4) {
  return arg4;
}
void* func83(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4.member0;
}
float func84(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4.member1;
}
uint8_t func85(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4.member2;
}
float func86(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  return arg4.member0;
}
double func87(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  return arg4.member1;
}
float func88(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member0;
}
double func89(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member1;
}
void* func90(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member2;
}
uint8_t func91(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member3;
}
uint8_t func92(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member4[0];
}
uint8_t func93(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member4[1];
}
uint8_t func94(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member4[2];
}
char func95(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, char arg5) {
  return arg5;
}
short int func96(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, short int arg5) {
  return arg5;
}
int func97(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, int arg5) {
  return arg5;
}
long long int func98(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, long long int arg5) {
  return arg5;
}
void* func99(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, void* arg5) {
  return arg5;
}
float func100(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, float arg5) {
  return arg5;
}
double func101(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, double arg5) {
  return arg5;
}
void* func102(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5.member0;
}
float func103(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5.member1;
}
uint8_t func104(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5.member2;
}
float func105(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  return arg5.member0;
}
double func106(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  return arg5.member1;
}
float func107(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member0;
}
double func108(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member1;
}
void* func109(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member2;
}
uint8_t func110(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member3;
}
uint8_t func111(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member4[0];
}
uint8_t func112(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member4[1];
}
uint8_t func113(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member4[2];
}
char func114(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, char arg6) {
  return arg6;
}
short int func115(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, short int arg6) {
  return arg6;
}
int func116(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, int arg6) {
  return arg6;
}
long long int func117(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, long long int arg6) {
  return arg6;
}
void* func118(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, void* arg6) {
  return arg6;
}
float func119(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, float arg6) {
  return arg6;
}
double func120(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, double arg6) {
  return arg6;
}
void* func121(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6.member0;
}
float func122(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6.member1;
}
uint8_t func123(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6.member2;
}
float func124(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  return arg6.member0;
}
double func125(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  return arg6.member1;
}
float func126(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member0;
}
double func127(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member1;
}
void* func128(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member2;
}
uint8_t func129(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member3;
}
uint8_t func130(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member4[0];
}
uint8_t func131(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member4[1];
}
uint8_t func132(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member4[2];
}
