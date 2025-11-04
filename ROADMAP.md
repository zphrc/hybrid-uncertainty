## **Phase 1 — Environment & Pipeline Setup**

- [x] Set up project directory
- [x] Implemented training (`hu-train`), scoring (`hu-score`), and evaluation (`hu-evaluate`) CLIs  
- [x] Integrated entropy, gradient, and hybrid danger score computation  
- [x] MNIST + ShuffleNetV2 - Train, score, and evaluate

---

## **Phase 2 — Deeper Analysis**

**Goal:** Evaluate multiple dataset-model pairs and compare hybrid vs. individual metrics.

- [ ] MNIST + MobileNetV3-Small — Train, score, and evaluate
- [ ] Fashion-MNIST + ShuffleNetV2 — Train, score, and evaluate
- [ ] Fashion-MNIST + MobileNetV3-Small — Train, score, and evaluate
- [ ] CIFAR-10 + MobileNetV3-Small — Train, score, and evaluate
- [ ] CIFAR-10 + EfficientNetV2-S — Train, score, and evaluate
- [ ] Metric comparison — Compare entropy-only, gradient-only, and hybrid results  
- [ ] Statistical validation — Bootstrap confidence intervals & ARC AUC  
- [ ] Summarize findings — Prepare results tables and plots for Chapter 4

---

## **Phase 3 — Prototype Development**

**Goal:** Implement an interactive prototype for demonstrating uncertainty evaluation.

- [ ] Streamlit frontend — Build interface for image upload and result visualization  
- [ ] FastAPI backend — Implement API for inference and uncertainty scoring  
- [ ] Integration testing — Connect frontend and backend for local prototype demo  
- [ ] Demo materials — Prepare screenshots, videos, and presentation dataset  
