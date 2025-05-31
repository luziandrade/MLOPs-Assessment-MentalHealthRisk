# MLOPs-Assessment-MentalHealthRisk

The goal of this project is to build an end-to-end **MLOps system** that supports the full machine learning lifecycle — from training to deployment and monitoring — with deployments to both **virtual machines (VMs)** and **Kubernetes clusters via Minikube**.

---

## 📌 Branching Strategy
graph TD
    A(main)
    A --> B(dev)
    B --> C(RB-1)
    C --> D(FB-A)
    C --> E(FB-B)
    C -->|PR| B
    B -->|Deploy to Dev| A
    A -->|Deploy to Prod| A
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

### 📈 Continuous Training - Continuous Monitoring (CT/CM)
graph TD
    A[Push to Feature Branch / PR to dev/main] --> B(CI: Build & Test)
    B --> C{Branch}
    C -- dev --> D[CD: Deploy to Dev]
    C -- main --> E[Manual Promote to Prod]

    subgraph CT [Continuous Training]
        F[Trigger: Schedule / Manual / Alert]
        F --> G[Retrain Model (retrain_model.py)]
        G --> H[Upload model.pkl Artifact]
        H --> I[Trigger CI/CD Workflow]
    end

    subgraph CM [Continuous Monitoring]
        J[Monitor Predictions / Metrics / Drift]
        J --> |Degradation| F
    end
