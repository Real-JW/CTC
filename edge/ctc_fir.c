#include "ctc_fir.h"

#include <string.h>

static int16_t saturate_q15(int64_t value) {
    value >>= MODEL2CTC_COEFF_Q;
    if (value > 32767) return 32767;
    if (value < -32768) return -32768;
    return (int16_t)value;
}

void model2ctc_fir_init(model2ctc_fir_t *state) {
    memset(state, 0, sizeof(*state));
}

void model2ctc_fir_process(
    model2ctc_fir_t *state,
    const int16_t input[2],
    int16_t output[2]
) {
    state->history[0][state->write_index] = input[0];
    state->history[1][state->write_index] = input[1];

    for (uint16_t out = 0; out < 2; ++out) {
        int64_t accumulator = 0;
        for (uint16_t in = 0; in < 2; ++in) {
            uint16_t history_index = state->write_index;
            for (uint16_t tap = 0; tap < MODEL2CTC_TAPS; ++tap) {
                accumulator += (int32_t)model2ctc_coeffs[out][in][tap]
                    * state->history[in][history_index];
                history_index = history_index == 0 ? MODEL2CTC_TAPS - 1 : history_index - 1;
            }
        }
        output[out] = saturate_q15(accumulator);
    }

    state->write_index += 1;
    if (state->write_index == MODEL2CTC_TAPS) state->write_index = 0;
}
