# MLOPs-Assessment-MentalHealthRisk

The goal of this project is to build an end-to-end **MLOps system** that supports the full machine learning lifecycle — from training to deployment and monitoring — with deployments to both **virtual machines (VMs)** and **Kubernetes clusters via Minikube**.

---

## 📌 Branching Strategy
main
│
├── dev
│ │
│ ├── RB-1 ──┐
│ │ │
│ │ FB-A ──┐
│ │ │
│ │ FB-B ─┘
│ │
│ └── PR RB-1 → dev
│ (Deploy to Dev)
│
└── Merge dev → main
(Deploy to Prod)

- **FB**: Feature Branch  
- **RB**: Release Branch  
- PRs flow: `FB` → `RB-*` → `dev` → `main`  
- Deployments:
  - Dev environment: on merge to `dev`
  - Prod environment: on merge to `main`

---

## 📌 Workflow Strategy

### 🛠 Continuous Integration / Continuous Deployment (CI/CD)
- **CI**: Triggered on push to any feature branch and on PRs to `dev` and `main`
- **CD**:
  - Deploy to **dev** on merge to `dev`
  - Deploy to **prod** via manual promotion (e.g., `promote_docker_image.yml`)

---

### 📈 Continuous Training (CT)
Trigger: Scheduled / Manual / Alert
│
▼
Continuous Training Workflow
├─ Checkout code & setup environment
├─ Retrain model (retrain_model.py)
├─ Save updated model.pkl
└─ Upload model.pkl artifact
│
▼
Trigger CI/CD Workflow with new model
│
▼
Deploy to Dev (Manual promotion to Prod)
