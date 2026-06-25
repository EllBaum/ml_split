// bindings.cpp -- Python interface for ml_split (outgroup-based subtree merge).
//
//   import ml_split
//   r = ml_split.merge(newick_a, newick_b, interest_a, interest_b, msa,
//                      inherited_a=[...], inherited_b=[...], model="JTT")
//   r.newick, r.loglik
//
// msa is either a FASTA path (str) or a {name: sequence} dict. It must contain
// rows for realA U realB and for any inherited taxa (the split-edge case scores
// inherited clades, which needs their sequences).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "merge_session.h"
#include "msa.h"
#include "subst_model.h"
#include "likelihood_scorer.h"
#include "jtt.h"
#include "jck.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;

static MSA dict_to_msa_pair(const py::dict& da, const py::dict& db) {
    // Union by name; a taxon in both must carry identical sequence.
    std::vector<std::string> names, seqs;
    std::unordered_map<std::string, size_t> idx;
    auto add = [&](const py::dict& d) {
        for (auto kv : d) {
            std::string n = py::cast<std::string>(py::str(kv.first));
            std::string s = py::cast<std::string>(py::str(kv.second));
            auto it = idx.find(n);
            if (it == idx.end()) { idx[n] = names.size(); names.push_back(n); seqs.push_back(s); }
            else if (seqs[it->second] != s)
                throw std::invalid_argument("taxon '" + n + "' has different sequences in the two MSAs");
        }
    };
    add(da); add(db);
    return MSA::from_sequences(names, seqs, false);
}

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

    mod.def("score",
        [](const std::string& newick, py::object msa, std::string model) {
            MSA full = build_msa(msa);
            std::vector<double> tmp(1 << 16, 0.1);
            Tree t0 = Tree::from_newick(newick, tmp.data());
            std::vector<double> bl((size_t)t0.n_nodes * 3, 0.1);
            Tree t = Tree::from_newick(newick, bl.data(), t0.taxon_names);
            MSA sl = slice_msa(full, t.taxon_names);   // tree-order rows
            auto mdl = build_model(model, sl);
            LikelihoodScorer S(t, sl, *mdl, bl.data());
            return S.score();
        },
        py::arg("newick"), py::arg("msa"), py::arg("model") = "JTT",
        "Log-likelihood of `newick` evaluated at its own branch lengths, on the "
        "given alignment (path or dict) under `model`. No BL/topology optimization "
        "-- for cross-checking a fixed tree against e.g. raxml-ng --opt-branches off.");

    mod.def("merge",
        [](std::string newick_a, std::string newick_b,
           std::string interest_a, std::string interest_b,
           py::object msa, py::object msa_a, py::object msa_b,
           std::vector<std::vector<std::string>> inherited_a,
           std::vector<std::vector<std::string>> inherited_b,
           std::string model, int window, double connector_init,
           std::string full_blo, double eps_5blo, double eps_fulltree) {
            MSA full = (!msa.is_none())
                ? build_msa(msa)
                : (!msa_a.is_none() && !msa_b.is_none())
                    ? dict_to_msa_pair(msa_a.cast<py::dict>(), msa_b.cast<py::dict>())
                    : throw std::invalid_argument(
                          "provide either msa=... or both msa_a=... and msa_b=...");
            auto mdl = build_model(model, full);
            MergeInput in;
            in.newick_a   = std::move(newick_a);   in.newick_b   = std::move(newick_b);
            in.interest_a = std::move(interest_a); in.interest_b = std::move(interest_b);
            in.inherited_a = std::move(inherited_a); in.inherited_b = std::move(inherited_b);
            return run_merge(in, full, *mdl, window, connector_init, full_blo,
                             eps_5blo, eps_fulltree);
        },
        py::arg("newick_a"), py::arg("newick_b"),
        py::arg("interest_a"), py::arg("interest_b"),
        py::arg("msa") = py::none(),
        py::arg("msa_a") = py::none(),
        py::arg("msa_b") = py::none(),
        py::arg("inherited_a") = std::vector<std::vector<std::string>>{},
        py::arg("inherited_b") = std::vector<std::vector<std::string>>{},
        py::arg("model") = "JTT",
        py::arg("window") = 14,
        py::arg("connector_init") = 0.1,
        py::arg("full_blo") = "off",
        py::arg("eps_5blo") = 1000.0,
        py::arg("eps_fulltree") = -1.0,
        "Merge two subtrees. Alignment as msa=(path|dict) OR msa_a=dict, msa_b=dict "
        "(merged by name). inherited_a/b are lists of clades (lists of leaf names).");
}