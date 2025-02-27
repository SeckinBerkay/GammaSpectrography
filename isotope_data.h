// isotope_data.c

// Because these are "const", they are stored in read-only memory (Flash)
// on most embedded toolchains, saving precious RAM.
// isotope_data.h
#include <stdint.h>



#define MAX_LINES_PER_ISOTOPE 12

typedef struct {
    const char *name;
    uint16_t energies[MAX_LINES_PER_ISOTOPE];
    uint8_t num_energies;
} IsotopeDef;

// Declare them as extern so other files can use them



IsotopeDef g_isotopes[] = {
    { "Am-241", {59}, 1 },
    { "Ba-133", {356, 80}, 2 },
    { "Co-57", {122}, 1 },
    { "Co-60", {1173, 1332}, 2 },
    { "Cs-137", {662}, 1 },
    { "Eu-152",  {121, 344, 244, 411, 444, 779, 867, 964, 1086, 1089, 1112, 1408}, 12 },
    { "K-40",  {511, 1460}, 2 },
    { "Na-22",  {511, 1274}, 2 },
    { "Ra-226",  {186}, 1 },
    { "Th-232",  {63, 140}, 2 }
    // etc...
};

// Optionally, keep track of how many isotopes:
const uint8_t g_num_isotopes = 10;//sizeof(g_isotopes) / sizeof(g_isotopes[0]);
