#ifndef MODEL2CTC_FIR_H
#define MODEL2CTC_FIR_H

#include <stdint.h>
#include "model2ctc_coefficients.h"

typedef struct {
    int16_t history[2][MODEL2CTC_TAPS];
    uint16_t write_index;
} model2ctc_fir_t;

void model2ctc_fir_init(model2ctc_fir_t *state);
void model2ctc_fir_process(
    model2ctc_fir_t *state,
    const int16_t input[2],
    int16_t output[2]
);

#endif
