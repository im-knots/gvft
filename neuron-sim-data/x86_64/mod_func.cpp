#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _Acetylcholine_reg(void);
extern void _Generic_GJ_reg(void);
extern void _Glutamate_reg(void);
extern void _LeakConductance_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"Acetylcholine.mod\"");
    fprintf(stderr, " \"Generic_GJ.mod\"");
    fprintf(stderr, " \"Glutamate.mod\"");
    fprintf(stderr, " \"LeakConductance.mod\"");
    fprintf(stderr, "\n");
  }
  _Acetylcholine_reg();
  _Generic_GJ_reg();
  _Glutamate_reg();
  _LeakConductance_reg();
}

#if defined(__cplusplus)
}
#endif
