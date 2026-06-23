// jtt.h - Jones-Taylor-Thornton (1992) amino-acid model.

#pragma once

#include "empirical_aa_model.h"

// JTT: Jones, Taylor & Thornton (1992). AA order: ACDEFGHIKLMNPQRSTVWY.
class JTT : public EmpiricalAAModel {
public:
    JTT();
};
