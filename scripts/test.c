#include <stdint.h>

// [limitation--no-relocatable] due to a known limitation, we can't process
// relocatable executables (we can't run relocations at all).
// adding main() allows us to compile this to a 'full' executable rather than a .o
int main() {}


char func000(char arg0) {
  return arg0;
}

short int func001(short int arg0) {
  return arg0;
}

int func002(int arg0) {
  return arg0;
}

long long int func003(long long int arg0) {
  return arg0;
}

void* func004(void* arg0) {
  return arg0;
}

float func005(float arg0) {
  return arg0;
}

double func006(double arg0) {
  return arg0;
}
struct small {
  void* member0;
  float member1;
  uint8_t member2;
};
struct small func007(struct small arg0) {
  return arg0;
}
void* func008(struct small arg0) {
  return arg0.member0;
}
float func009(struct small arg0) {
  return arg0.member1;
}
uint8_t func010(struct small arg0) {
  return arg0.member2;
}
struct small_xmms {
  float member0;
  double member1;
};
struct small_xmms func011(struct small_xmms arg0) {
  return arg0;
}
float func012(struct small_xmms arg0) {
  return arg0.member0;
}
double func013(struct small_xmms arg0) {
  return arg0.member1;
}
struct big {
  float member0;
  double member1;
  void* member2;
  uint8_t member3;
  uint8_t member4[3];
};
struct big func014(struct big arg0) {
  return arg0;
}
float func015(struct big arg0) {
  return arg0.member0;
}
double func016(struct big arg0) {
  return arg0.member1;
}
void* func017(struct big arg0) {
  return arg0.member2;
}
uint8_t func018(struct big arg0) {
  return arg0.member3;
}
uint8_t func019(struct big arg0) {
  return arg0.member4[0];
}
uint8_t func020(struct big arg0) {
  return arg0.member4[1];
}
uint8_t func021(struct big arg0) {
  return arg0.member4[2];
}
char func022(struct small arg0, char arg1) {
  return arg1;
}
short int func023(struct small arg0, short int arg1) {
  return arg1;
}
int func024(struct small arg0, int arg1) {
  return arg1;
}
long long int func025(struct small arg0, long long int arg1) {
  return arg1;
}
void* func026(struct small arg0, void* arg1) {
  return arg1;
}
float func027(struct small arg0, float arg1) {
  return arg1;
}
double func028(struct small arg0, double arg1) {
  return arg1;
}
struct small func029(struct small arg0, struct small arg1) {
  return arg1;
}
void* func030(struct small arg0, struct small arg1) {
  return arg1.member0;
}
float func031(struct small arg0, struct small arg1) {
  return arg1.member1;
}
uint8_t func032(struct small arg0, struct small arg1) {
  return arg1.member2;
}
struct small_xmms func033(struct small arg0, struct small_xmms arg1) {
  return arg1;
}
float func034(struct small arg0, struct small_xmms arg1) {
  return arg1.member0;
}
double func035(struct small arg0, struct small_xmms arg1) {
  return arg1.member1;
}
struct big func036(struct small arg0, struct big arg1) {
  return arg1;
}
float func037(struct small arg0, struct big arg1) {
  return arg1.member0;
}
double func038(struct small arg0, struct big arg1) {
  return arg1.member1;
}
void* func039(struct small arg0, struct big arg1) {
  return arg1.member2;
}
uint8_t func040(struct small arg0, struct big arg1) {
  return arg1.member3;
}
uint8_t func041(struct small arg0, struct big arg1) {
  return arg1.member4[0];
}
uint8_t func042(struct small arg0, struct big arg1) {
  return arg1.member4[1];
}
uint8_t func043(struct small arg0, struct big arg1) {
  return arg1.member4[2];
}
char func044(void* arg0, struct small arg1, char arg2) {
  return arg2;
}
short int func045(void* arg0, struct small arg1, short int arg2) {
  return arg2;
}
int func046(void* arg0, struct small arg1, int arg2) {
  return arg2;
}
long long int func047(void* arg0, struct small arg1, long long int arg2) {
  return arg2;
}
void* func048(void* arg0, struct small arg1, void* arg2) {
  return arg2;
}
float func049(void* arg0, struct small arg1, float arg2) {
  return arg2;
}
double func050(void* arg0, struct small arg1, double arg2) {
  return arg2;
}
struct small func051(void* arg0, struct small arg1, struct small arg2) {
  return arg2;
}
void* func052(void* arg0, struct small arg1, struct small arg2) {
  return arg2.member0;
}
float func053(void* arg0, struct small arg1, struct small arg2) {
  return arg2.member1;
}
uint8_t func054(void* arg0, struct small arg1, struct small arg2) {
  return arg2.member2;
}
struct small_xmms func055(void* arg0, struct small arg1, struct small_xmms arg2) {
  return arg2;
}
float func056(void* arg0, struct small arg1, struct small_xmms arg2) {
  return arg2.member0;
}
double func057(void* arg0, struct small arg1, struct small_xmms arg2) {
  return arg2.member1;
}
struct big func058(void* arg0, struct small arg1, struct big arg2) {
  return arg2;
}
float func059(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member0;
}
double func060(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member1;
}
void* func061(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member2;
}
uint8_t func062(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member3;
}
uint8_t func063(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member4[0];
}
uint8_t func064(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member4[1];
}
uint8_t func065(void* arg0, struct small arg1, struct big arg2) {
  return arg2.member4[2];
}
char func066(void* arg0, void* arg1, struct small arg2, char arg3) {
  return arg3;
}
short int func067(void* arg0, void* arg1, struct small arg2, short int arg3) {
  return arg3;
}
int func068(void* arg0, void* arg1, struct small arg2, int arg3) {
  return arg3;
}
long long int func069(void* arg0, void* arg1, struct small arg2, long long int arg3) {
  return arg3;
}
void* func070(void* arg0, void* arg1, struct small arg2, void* arg3) {
  return arg3;
}
float func071(void* arg0, void* arg1, struct small arg2, float arg3) {
  return arg3;
}
double func072(void* arg0, void* arg1, struct small arg2, double arg3) {
  return arg3;
}
struct small func073(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3;
}
void* func074(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3.member0;
}
float func075(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3.member1;
}
uint8_t func076(void* arg0, void* arg1, struct small arg2, struct small arg3) {
  return arg3.member2;
}
struct small_xmms func077(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  return arg3;
}
float func078(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  return arg3.member0;
}
double func079(void* arg0, void* arg1, struct small arg2, struct small_xmms arg3) {
  return arg3.member1;
}
struct big func080(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3;
}
float func081(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member0;
}
double func082(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member1;
}
void* func083(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member2;
}
uint8_t func084(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member3;
}
uint8_t func085(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member4[0];
}
uint8_t func086(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member4[1];
}
uint8_t func087(void* arg0, void* arg1, struct small arg2, struct big arg3) {
  return arg3.member4[2];
}
char func088(void* arg0, void* arg1, void* arg2, struct small arg3, char arg4) {
  return arg4;
}
short int func089(void* arg0, void* arg1, void* arg2, struct small arg3, short int arg4) {
  return arg4;
}
int func090(void* arg0, void* arg1, void* arg2, struct small arg3, int arg4) {
  return arg4;
}
long long int func091(void* arg0, void* arg1, void* arg2, struct small arg3, long long int arg4) {
  return arg4;
}
void* func092(void* arg0, void* arg1, void* arg2, struct small arg3, void* arg4) {
  return arg4;
}
float func093(void* arg0, void* arg1, void* arg2, struct small arg3, float arg4) {
  return arg4;
}
double func094(void* arg0, void* arg1, void* arg2, struct small arg3, double arg4) {
  return arg4;
}
struct small func095(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4;
}
void* func096(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4.member0;
}
float func097(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4.member1;
}
uint8_t func098(void* arg0, void* arg1, void* arg2, struct small arg3, struct small arg4) {
  return arg4.member2;
}
struct small_xmms func099(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  return arg4;
}
float func100(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  return arg4.member0;
}
double func101(void* arg0, void* arg1, void* arg2, struct small arg3, struct small_xmms arg4) {
  return arg4.member1;
}
struct big func102(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4;
}
float func103(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member0;
}
double func104(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member1;
}
void* func105(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member2;
}
uint8_t func106(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member3;
}
uint8_t func107(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member4[0];
}
uint8_t func108(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member4[1];
}
uint8_t func109(void* arg0, void* arg1, void* arg2, struct small arg3, struct big arg4) {
  return arg4.member4[2];
}
char func110(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, char arg5) {
  return arg5;
}
short int func111(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, short int arg5) {
  return arg5;
}
int func112(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, int arg5) {
  return arg5;
}
long long int func113(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, long long int arg5) {
  return arg5;
}
void* func114(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, void* arg5) {
  return arg5;
}
float func115(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, float arg5) {
  return arg5;
}
double func116(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, double arg5) {
  return arg5;
}
struct small func117(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5;
}
void* func118(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5.member0;
}
float func119(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5.member1;
}
uint8_t func120(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small arg5) {
  return arg5.member2;
}
struct small_xmms func121(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  return arg5;
}
float func122(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  return arg5.member0;
}
double func123(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct small_xmms arg5) {
  return arg5.member1;
}
struct big func124(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5;
}
float func125(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member0;
}
double func126(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member1;
}
void* func127(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member2;
}
uint8_t func128(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member3;
}
uint8_t func129(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member4[0];
}
uint8_t func130(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member4[1];
}
uint8_t func131(void* arg0, void* arg1, void* arg2, void* arg3, struct small arg4, struct big arg5) {
  return arg5.member4[2];
}
char func132(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, char arg6) {
  return arg6;
}
short int func133(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, short int arg6) {
  return arg6;
}
int func134(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, int arg6) {
  return arg6;
}
long long int func135(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, long long int arg6) {
  return arg6;
}
void* func136(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, void* arg6) {
  return arg6;
}
float func137(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, float arg6) {
  return arg6;
}
double func138(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, double arg6) {
  return arg6;
}
struct small func139(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6;
}
void* func140(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6.member0;
}
float func141(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6.member1;
}
uint8_t func142(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small arg6) {
  return arg6.member2;
}
struct small_xmms func143(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  return arg6;
}
float func144(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  return arg6.member0;
}
double func145(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct small_xmms arg6) {
  return arg6.member1;
}
struct big func146(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6;
}
float func147(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member0;
}
double func148(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member1;
}
void* func149(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member2;
}
uint8_t func150(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member3;
}
uint8_t func151(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member4[0];
}
uint8_t func152(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member4[1];
}
uint8_t func153(void* arg0, void* arg1, void* arg2, void* arg3, void* arg4, struct small arg5, struct big arg6) {
  return arg6.member4[2];
}
