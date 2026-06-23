// bindings.cpp — Python interface for ml_split (outgroup-based subtree merge).
//
//   import ml_split
//   r = ml_split.merge(newick_a, newick_b, interest_a, interest_b, msa,
//                      inherited_a=[...], inherited_b=[...], model="JTT")
//   r.newick, r.loglik
//
// msa is either a FASTA path (str) or a {name: sequence} dict. It must contain
// rows for realA ∪ realB and for any inherited taxa (the split-edge case scores
// inherited clades, which needs their sequences).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "merge_session.h"
#include "msa.h"
#include "subst_model.h"
#include "jtt.h"
#include "jck.h"
#include <memory>
#include <stdexcept>

namespace py = pybind11;

static MSA build_msa(const py::object& msa) {
    if (py::isinstance<py::str>(msa))
        return MSA::from_fasta(msa.cast<std::string>(), false);
    if (py::isinstance<py::dict>(msa)) {
        std::vector<std::string> names, seqs;
        for (auto kv : msa.cast<py::dict>()) {
            names.push_back(py::cast<std::string>(py::str(kv.first)));
            seqs.push_back(py::cast<std::string>(py::str(kv.second)));
        }
        return MSA::from_sequences(names, seqs, false);
    }
    throw std::invalid_argument("msa must be a FASTA path (str) or a {name: seq} dict");
}

static std::unique_ptr<SubstModel> build_model(const std::string& m, const MSA& full) {
    if (m == "JTT") return std::make_unique<JTT>();
    if (m == "JC" || m == "JCK") {
        int k = (full.seq_type == SeqType::AA) ? 20 : 4;
        return std::make_unique<JCK>(k);
    }
    throw std::invalid_argument("unknown model '" + m + "' (supported: JTT, JC)");
}

PYBIND11_MODULE(ml_split, mod) {
    mod.doc() = "Outgroup-based subtree merge.";

    py::class_<MergeResult>(mod, "MergeResult")
        .def_readonly("newick", &MergeResult::newick)
        .def_readonly("loglik", &MergeResult::loglik)
        .def("__repr__", [](const MergeResult& r) {
            return "<MergeResult loglik=" + std::to_string(r.loglik) + ">";
        });

    mod.def("merge",
        [](std::string newick_a, std::string newick_b,
           std::string interest_a, std::string interest_b,
           py::object msa,
           std::vector<std::vector<std::string>> inherited_a,
           std::vector<std::vector<std::string>> inherited_b,
           std::string model, int window, double connector_init) {
            MSA full = build_msa(msa);
            auto mdl = build_model(model, full);
            MergeInput in;
            in.newick_a   = std::move(newick_a);   in.newick_b   = std::move(newick_b);
            in.interest_a = std::move(interest_a); in.interest_b = std::move(interest_b);
            in.inherited_a = std::move(inherited_a); in.inherited_b = std::move(inherited_b);
            return run_merge(in, full, *mdl, window, connector_init);
        },
        py::arg("newick_a"), py::arg("newick_b"),
        py::arg("interest_a"), py::arg("interest_b"),
        py::arg("msa"),
        py::arg("inherited_a") = std::vector<std::vector<std::string>>{},
        py::arg("inherited_b") = std::vector<std::vector<std::string>>{},
        py::arg("model") = "JTT",
        py::arg("window") = 14,
        py::arg("connector_init") = 0.1,
        "Merge two subtrees at the outgroup-of-interest windows; returns "
        "MergeResult(newick, loglik). inherited_a/b are lists of clades "
        "(each a list of leaf names). msa is a FASTA path or {name: seq} dict.");
}
