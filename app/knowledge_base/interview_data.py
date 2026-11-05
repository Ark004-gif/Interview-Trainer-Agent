"""
Interview Knowledge Base
Role-specific questions, behavioral scenarios, and HR guidelines
used to seed the ChromaDB vector store for RAG retrieval.
"""

# ── Technical Questions by Role ───────────────────────────────────────────────
TECHNICAL_QUESTIONS = {
    "Software Engineer": [
        {
            "question": "Explain the difference between a process and a thread.",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "A process is an independent program with its own memory space, while a thread is "
                "a lightweight unit of execution that shares memory with other threads in the same process. "
                "Processes are isolated from each other and communicate via IPC mechanisms like pipes or sockets. "
                "Threads within a process share heap memory, file handles, and code, making communication faster "
                "but requiring synchronization (locks/semaphores) to prevent race conditions."
            ),
            "tips": "Use concrete examples: a browser where each tab is a process, and rendering/JS engine are threads.",
        },
        {
            "question": "What is Big-O notation? Explain O(n log n) with an example.",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Big-O describes the upper-bound growth rate of an algorithm's time/space complexity "
                "relative to input size n. O(n log n) means for n elements, operations grow as n multiplied by log n. "
                "Merge Sort is a classic O(n log n) example: it divides the array into halves (log n levels) "
                "and merges n elements at each level."
            ),
            "tips": "Draw the recursion tree for merge sort to visually explain the log n levels.",
        },
        {
            "question": "Describe SOLID principles with real-world examples.",
            "category": "technical",
            "difficulty": "senior",
            "model_answer": (
                "SOLID stands for: Single Responsibility (a class does one thing — e.g., UserService only handles user logic), "
                "Open/Closed (open for extension, closed for modification — e.g., adding payment methods via an interface), "
                "Liskov Substitution (subclasses must be replaceable by parent — e.g., Square shouldn't extend Rectangle if it breaks area logic), "
                "Interface Segregation (no client forced to depend on unused methods), "
                "Dependency Inversion (depend on abstractions, not concretions — use DI containers)."
            ),
            "tips": "Give one concrete violation you fixed in a real project; it shows depth.",
        },
        {
            "question": "How does garbage collection work in Java/Python?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Java uses generational GC (Young/Old/PermGen). Objects are first allocated in Eden space; "
                "survivors are promoted to Survivor then Old generation. The JVM uses Mark-and-Sweep with Compact. "
                "Python uses reference counting + a cyclic GC for circular references. When ref count drops to 0, memory is freed immediately."
            ),
            "tips": "Mention GC tuning flags in Java (-Xmx, -XX:+UseG1GC) to show practical knowledge.",
        },
        {
            "question": "What is a deadlock? How do you prevent it?",
            "category": "technical",
            "difficulty": "senior",
            "model_answer": (
                "Deadlock occurs when two or more threads wait for each other's locked resources indefinitely. "
                "Four conditions must hold: mutual exclusion, hold-and-wait, no preemption, circular wait. "
                "Prevention: impose a consistent lock ordering, use tryLock with timeout, use lock-free data structures, "
                "or detect via timeout and rollback."
            ),
            "tips": "Show a simple 2-thread example with pseudocode to make the circular wait concrete.",
        },
        {
            "question": "Explain RESTful API design best practices.",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Use nouns for resources (/users, /orders), HTTP verbs for actions (GET/POST/PUT/DELETE), "
                "proper status codes (200 OK, 201 Created, 404 Not Found, 422 Unprocessable), versioning (/api/v1/), "
                "pagination (limit/offset or cursor-based), statelessness, and HATEOAS for discoverability. "
                "Secure with HTTPS, JWT/OAuth2, and rate limiting."
            ),
            "tips": "Mention OpenAPI/Swagger for documentation as a sign of production-readiness.",
        },
        {
            "question": "How would you design a URL shortener like bit.ly?",
            "category": "system_design",
            "difficulty": "senior",
            "model_answer": (
                "Core components: API server (POST /shorten → short code), Base62 encoding of an auto-increment ID, "
                "Redis cache for hot URLs, read-heavy relational DB (PostgreSQL) for persistence. "
                "Scale: CDN for redirect latency, DB sharding by short code hash, async analytics pipeline via Kafka."
            ),
            "tips": "Start with back-of-envelope estimates (100M URLs, 10B redirects/day) before diving into design.",
        },
    ],
    "Data Scientist": [
        {
            "question": "Explain the bias-variance tradeoff.",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Bias is error from wrong assumptions (underfitting); variance is error from sensitivity to training noise (overfitting). "
                "High bias → model too simple (e.g., linear for non-linear data). High variance → model memorizes training data. "
                "Tradeoff: as model complexity increases, bias decreases but variance increases. "
                "Regularization (L1/L2), cross-validation, and ensemble methods help balance both."
            ),
            "tips": "Draw the classic U-shaped total error curve to visualize the sweet spot.",
        },
        {
            "question": "What is the difference between bagging and boosting?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Bagging (Bootstrap Aggregating): trains models in parallel on random subsets, reduces variance (e.g., Random Forest). "
                "Boosting: trains models sequentially where each corrects predecessor's errors, reduces bias (e.g., XGBoost, AdaBoost). "
                "Bagging suits high-variance models (deep trees); boosting suits high-bias models."
            ),
            "tips": "Compare Random Forest vs. XGBoost performance on a tabular dataset you've worked with.",
        },
        {
            "question": "How do you handle class imbalance in a dataset?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Techniques: oversampling minority class (SMOTE), undersampling majority class, class_weight='balanced' in sklearn, "
                "threshold tuning on predicted probabilities, use Precision-Recall AUC instead of ROC-AUC, "
                "ensemble methods like BalancedRandomForest."
            ),
            "tips": "Always state the business context: false negatives may cost more than false positives (e.g., fraud detection).",
        },
        {
            "question": "Explain how gradient descent works.",
            "category": "technical",
            "difficulty": "junior",
            "model_answer": (
                "Gradient descent minimizes a loss function by iteratively updating parameters in the direction of the negative gradient. "
                "θ = θ − α·∇L(θ). Variants: Batch GD (full dataset), SGD (one sample), Mini-batch GD (subset). "
                "Adaptive methods (Adam, RMSProp) adjust the learning rate per parameter."
            ),
            "tips": "Explain learning rate sensitivity with the analogy of walking down a hill in the dark.",
        },
        {
            "question": "What is feature engineering and why does it matter?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Feature engineering transforms raw data into informative inputs for models. Techniques: "
                "log transforms for skewed distributions, interaction terms, polynomial features, target encoding, "
                "date decomposition (hour/day/weekday), lag features for time series. "
                "Good features often outperform complex models on same data."
            ),
            "tips": "Give a concrete example: extracting 'days since last purchase' from a transaction date for churn prediction.",
        },
    ],
    "Product Manager": [
        {
            "question": "How do you prioritize features in a product backlog?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Use frameworks like RICE (Reach × Impact × Confidence / Effort), MoSCoW (Must/Should/Could/Won't), "
                "or Kano model (basic, performance, delighters). "
                "Always tie to OKRs/business goals, gather data from customer interviews, NPS, analytics, "
                "and balance short-term revenue vs. long-term platform health."
            ),
            "tips": "Mention a real scenario where you de-prioritized a high-effort, low-impact feature.",
        },
        {
            "question": "How would you define and measure product success?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Define success via North Star Metric (e.g., weekly active users, GMV) aligned to business outcome. "
                "Use the AARRR funnel (Acquisition, Activation, Retention, Revenue, Referral). "
                "Set leading indicators (feature adoption rate) alongside lagging indicators (revenue)."
            ),
            "tips": "Avoid vanity metrics. Show you understand the difference between output metrics and outcome metrics.",
        },
        {
            "question": "Describe how you handle conflicting stakeholder requirements.",
            "category": "behavioral",
            "difficulty": "senior",
            "model_answer": (
                "Acknowledge each stakeholder's perspective, map requirements to shared business goals, "
                "use data (user research, A/B tests, revenue projections) to facilitate objective decisions, "
                "escalate with options (not problems), and document trade-offs transparently."
            ),
            "tips": "Structure your answer with the STAR method: describe the conflict, your mediation, and the outcome.",
        },
    ],
    "DevOps Engineer": [
        {
            "question": "Explain the CI/CD pipeline you have designed or worked with.",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "A CI/CD pipeline typically includes: Source (Git push triggers pipeline), Build (compile/package), "
                "Test (unit, integration, SAST), Artifact (Docker image pushed to registry), "
                "Deploy (Helm chart to K8s via ArgoCD/Spinnaker), Post-deploy (smoke tests, canary monitoring, rollback gates)."
            ),
            "tips": "Name specific tools: GitHub Actions, Jenkins, GitLab CI, ArgoCD, Terraform.",
        },
        {
            "question": "What is Kubernetes and how does it handle container orchestration?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Kubernetes is an open-source container orchestration platform. Key concepts: "
                "Pod (smallest deployable unit), Deployment (manages ReplicaSets for scaling/rollout), "
                "Service (stable network endpoint), Ingress (HTTP routing), ConfigMap/Secret, "
                "HPA (Horizontal Pod Autoscaler). Scheduler places pods on nodes based on resource requests."
            ),
            "tips": "Walk through what happens from kubectl apply to a pod running to show end-to-end knowledge.",
        },
        {
            "question": "How do you approach infrastructure as code (IaC)?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "IaC (Terraform, Pulumi, CloudFormation) provisions infra declaratively, enabling version control, "
                "peer review, and reproducibility. Best practices: remote state (S3 + DynamoDB lock), "
                "modules for reusability, separate workspaces per environment, drift detection, "
                "and OPA/Sentinel for policy enforcement."
            ),
            "tips": "Explain a specific incident where IaC prevented a configuration drift issue.",
        },
    ],
    "ML Engineer": [
        {
            "question": "How do you deploy a machine learning model to production?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "Steps: Containerize model (Docker), expose via REST API (FastAPI/Flask) or gRPC, "
                "push to container registry, deploy to Kubernetes or serverless (Lambda), "
                "set up monitoring (data drift via Evidently, latency/error metrics in Prometheus/Grafana), "
                "implement A/B testing or shadow mode rollout, automate retraining pipeline."
            ),
            "tips": "Mention MLflow for experiment tracking and model registry to show ML lifecycle awareness.",
        },
        {
            "question": "What is model drift and how do you detect it?",
            "category": "technical",
            "difficulty": "senior",
            "model_answer": (
                "Model drift: degradation of model performance over time. "
                "Data drift — input feature distribution shifts (detect with PSI, KL divergence, Kolmogorov-Smirnov test). "
                "Concept drift — relationship between features and target changes. "
                "Monitor: track prediction distribution, set up alerts on accuracy/AUC on labeled incoming samples, use shadow models."
            ),
            "tips": "Reference real-world causes: seasonality, economic shifts, product changes.",
        },
    ],
    "Full Stack Developer": [
        {
            "question": "Explain the event loop in JavaScript.",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "JS is single-threaded. The event loop continuously checks the call stack and the task queue. "
                "Sync code runs on the call stack. Async callbacks (setTimeout, I/O) are pushed to the task queue. "
                "Microtasks (Promises, queueMicrotask) have higher priority than macrotasks and run before the next task."
            ),
            "tips": "Use Philip Roberts' JS Visualizer analogy: call stack, Web APIs, task queue, event loop.",
        },
        {
            "question": "How does React's virtual DOM work?",
            "category": "technical",
            "difficulty": "intermediate",
            "model_answer": (
                "React maintains a virtual DOM (lightweight JS object tree) mirroring the real DOM. "
                "On state/prop change, React creates a new virtual DOM tree, diffs it against the previous one (reconciliation), "
                "and applies minimal real DOM mutations (batched updates). This avoids expensive full re-renders."
            ),
            "tips": "Mention React Fiber (incremental rendering) and keys in lists as real optimizations.",
        },
    ],
    "Cloud Architect": [
        {
            "question": "Design a highly available, fault-tolerant architecture on IBM Cloud.",
            "category": "system_design",
            "difficulty": "senior",
            "model_answer": (
                "Use IBM Cloud multi-zone regions (MZR) with Load Balancer for HA. "
                "Compute: IBM Kubernetes Service (IKS) across 3 zones. "
                "Database: IBM Db2 with high-availability standbys or IBM Cloudant (geo-replicated). "
                "Storage: IBM COS (11 nines durability). Networking: VPC with security groups, "
                "IBM Cloud Internet Services (CIS) for DDoS and CDN. Monitoring: IBM Instana."
            ),
            "tips": "Calculate RPO/RTO targets first, then map architecture decisions to those requirements.",
        },
        {
            "question": "What is the difference between horizontal and vertical scaling?",
            "category": "technical",
            "difficulty": "junior",
            "model_answer": (
                "Vertical scaling: add more CPU/RAM to an existing server (scale up). Simple but has a ceiling and SPOF. "
                "Horizontal scaling: add more server instances behind a load balancer (scale out). "
                "Fault-tolerant, theoretically unlimited, but requires stateless design and distributed coordination."
            ),
            "tips": "Always pivot to stateless design being the prerequisite for horizontal scaling.",
        },
    ],
    "Business Analyst": [
        {
            "question": "How do you gather requirements from stakeholders?",
            "category": "technical",
            "difficulty": "junior",
            "model_answer": (
                "Techniques: stakeholder interviews (open-ended questions), workshops (JAD sessions), "
                "observation (shadowing users), document analysis (existing reports, process maps), "
                "surveys, and prototyping/wireframes for early feedback. "
                "Consolidate into BRD/FRS, validate with sign-off, and manage changes via a change log."
            ),
            "tips": "Mention the difference between stated needs and actual needs — stakeholders often describe solutions, not problems.",
        },
    ],
}

# ── Behavioral / HR Questions ────────────────────────────────────────────────
BEHAVIORAL_QUESTIONS = [
    {
        "question": "Tell me about yourself.",
        "category": "behavioral",
        "framework": "Present-Past-Future",
        "model_answer": (
            "Structure: Current role and key responsibilities (Present) → relevant prior experience (Past) "
            "→ why this role excites you and where you want to grow (Future). Keep it to 90 seconds. "
            "Quantify impact: 'Led a team of 5 and reduced deployment time by 40%' is stronger than 'managed deployments'."
        ),
        "tips": "Avoid reading your resume. Focus on narrative and energy — this sets the tone for the interview.",
    },
    {
        "question": "Describe a time you failed. What did you learn?",
        "category": "behavioral",
        "framework": "STAR",
        "model_answer": (
            "Situation: Pick a genuine, non-trivial failure (not 'I work too hard'). "
            "Task: What was your responsibility? "
            "Action: What did you do — including the mistake? "
            "Result: What happened, what was the impact, and crucially — what did you change afterward? "
            "Show self-awareness and growth mindset."
        ),
        "tips": "Interviewers want maturity, not perfection. A story with real consequences and real learning is compelling.",
    },
    {
        "question": "How do you handle working under pressure or tight deadlines?",
        "category": "behavioral",
        "framework": "STAR",
        "model_answer": (
            "Describe a specific high-pressure situation: prioritize ruthlessly using impact/effort matrix, "
            "communicate proactively with stakeholders on scope/timeline trade-offs, "
            "break the problem into manageable milestones, and use stress management techniques (deep work blocks, timeboxing). "
            "End with the positive outcome and what you'd do the same or differently."
        ),
        "tips": "Specificity wins. 'We had 48 hours to fix a payment processing bug affecting 50K users' beats vague answers.",
    },
    {
        "question": "Why do you want to work at this company?",
        "category": "behavioral",
        "framework": "Research-Align-Contribute",
        "model_answer": (
            "Research: Show you know the company's product, mission, recent news, tech stack, or culture. "
            "Align: Connect your past experience and values to what the company stands for. "
            "Contribute: State specifically what you can bring to the team and why now is the right time."
        ),
        "tips": "Never say 'compensation' or 'job security'. Link to mission, product, or growth opportunity.",
    },
    {
        "question": "Where do you see yourself in 5 years?",
        "category": "behavioral",
        "framework": "Growth-Role-Value",
        "model_answer": (
            "Show ambition without being threatening. Example: 'I want to deepen my expertise in distributed systems, "
            "take on technical leadership of a product area, and eventually mentor junior engineers. "
            "I see this role as the foundation for that path.' "
            "Align your growth with the company's trajectory."
        ),
        "tips": "Tailor to the seniority of the role. A junior should focus on learning; a senior on impact and leadership.",
    },
    {
        "question": "Tell me about a time you disagreed with a team member or manager.",
        "category": "behavioral",
        "framework": "STAR",
        "model_answer": (
            "Pick a professional disagreement (not personal conflict). Show: you listened to understand their position, "
            "you presented your view with data/logic (not emotion), you sought common ground, "
            "and you either changed your mind based on their reasoning or influenced the decision constructively. "
            "Outcome: describe what happened and that the relationship remained positive."
        ),
        "tips": "This tests emotional intelligence and professional maturity. Show respect even in disagreement.",
    },
    {
        "question": "How do you prioritize your work when you have multiple tasks?",
        "category": "behavioral",
        "framework": "Method-Example",
        "model_answer": (
            "Describe a system: Eisenhower Matrix (urgent/important quadrants), time-blocking, "
            "daily standups to surface blockers, clear definition of done. "
            "Give an example: 'When I had three concurrent deadlines, I mapped dependencies, identified the critical path, "
            "delegated testing tasks, and communicated the revised ETA to stakeholders 3 days early.'"
        ),
        "tips": "Show both system (how you generally operate) and adaptability (how you handle surprises).",
    },
    {
        "question": "What is your greatest strength and weakness?",
        "category": "behavioral",
        "framework": "Authentic-Evidence",
        "model_answer": (
            "Strength: Choose one that is genuinely relevant to the role and back it with a specific example and quantified result. "
            "Weakness: Choose a real weakness you are actively improving, describe the steps you are taking, "
            "and show progress. Avoid clichés like 'perfectionist' or 'I work too hard'."
        ),
        "tips": "Authenticity and self-awareness are the goals. An honest weakness with a growth plan impresses far more than a polished non-answer.",
    },
]

# ── Soft Skills Assessment Criteria ──────────────────────────────────────────
SOFT_SKILLS_CRITERIA = {
    "Communication": {
        "description": "Clarity, conciseness, active listening, non-verbal cues",
        "assessment_questions": [
            "How do you explain technical concepts to non-technical stakeholders?",
            "Describe a time you gave difficult feedback to a colleague.",
        ],
    },
    "Problem Solving": {
        "description": "Structured thinking, first-principles reasoning, creative solutions",
        "assessment_questions": [
            "Walk me through how you debug a complex production issue.",
            "Describe a problem you solved that had no obvious solution.",
        ],
    },
    "Teamwork & Collaboration": {
        "description": "Cross-functional collaboration, conflict resolution, knowledge sharing",
        "assessment_questions": [
            "Tell me about a successful cross-team project you contributed to.",
            "How do you onboard and support new team members?",
        ],
    },
    "Adaptability": {
        "description": "Handling ambiguity, embracing change, learning agility",
        "assessment_questions": [
            "Describe a time when project requirements changed significantly mid-way.",
            "How do you keep your skills current in a fast-changing field?",
        ],
    },
    "Leadership": {
        "description": "Influencing without authority, mentoring, driving outcomes",
        "assessment_questions": [
            "Tell me about a time you led without formal authority.",
            "How do you motivate team members who are disengaged?",
        ],
    },
}

# ── Industry Expectations by Experience Level ────────────────────────────────
EXPERIENCE_LEVEL_GUIDE = {
    "fresher": {
        "expectations": [
            "Strong CS fundamentals (DS&A, OS, DBMS, networking)",
            "At least 2 personal/academic projects with GitHub links",
            "Eagerness to learn and adaptability",
            "Basic understanding of the target technology stack",
        ],
        "focus_areas": ["data structures", "algorithms", "projects", "internship experience"],
        "salary_negotiation_tip": "Research market ranges (Glassdoor, LinkedIn). Highlight unique skills from projects.",
    },
    "junior": {
        "expectations": [
            "1–3 years of hands-on professional experience",
            "Ownership of small-to-medium features end-to-end",
            "Basic debugging and code review skills",
            "Familiarity with Agile/Scrum processes",
        ],
        "focus_areas": ["debugging", "code quality", "collaboration", "delivery"],
        "salary_negotiation_tip": "Anchor on your demonstrated impact, not just experience years.",
    },
    "mid-level": {
        "expectations": [
            "3–6 years of experience with measurable project impact",
            "Technical design capabilities and ability to mentor juniors",
            "Cross-team collaboration and stakeholder communication",
            "Performance optimization and production ownership",
        ],
        "focus_areas": ["system design basics", "mentorship", "ownership", "impact metrics"],
        "salary_negotiation_tip": "Quantify your impact in dollar or percentage terms before negotiating.",
    },
    "senior": {
        "expectations": [
            "6–10 years of experience with architectural decision-making",
            "Leading teams or technical domains",
            "Defining coding standards and technical roadmaps",
            "Cross-functional influence and executive communication",
        ],
        "focus_areas": ["system design", "leadership", "architecture", "strategy"],
        "salary_negotiation_tip": "Negotiate the full package: equity, bonus, PTO, remote flexibility — not just base salary.",
    },
    "lead": {
        "expectations": [
            "Ownership of a product area or engineering pillar",
            "Hiring, growing, and retaining engineers",
            "Driving multi-quarter technical initiatives",
            "Balancing technical excellence with delivery speed",
        ],
        "focus_areas": ["people management", "roadmap ownership", "technical vision"],
        "salary_negotiation_tip": "Frame your value as organizational leverage — every junior you level up multiplies output.",
    },
    "principal": {
        "expectations": [
            "Company-wide technical influence",
            "Defining architectural standards and long-term technology bets",
            "External thought leadership (papers, talks, OSS)",
            "Deep domain expertise recognized industry-wide",
        ],
        "focus_areas": ["org-level impact", "technical strategy", "thought leadership"],
        "salary_negotiation_tip": "At this level, total compensation (equity, signing bonus, retention) matters more than base salary.",
    },
}

# ── HR Guidelines ─────────────────────────────────────────────────────────────
HR_GUIDELINES = {
    "dos": [
        "Research the company thoroughly: mission, products, recent news, culture",
        "Prepare 3–5 stories using the STAR framework for behavioral questions",
        "Quantify achievements: percentages, dollar amounts, user counts, time saved",
        "Bring copies of your resume and a portfolio/GitHub link",
        "Prepare 3–5 thoughtful questions for your interviewers",
        "Send a thank-you email within 24 hours after the interview",
        "Dress appropriately (research the company culture for guidance)",
        "Arrive 10 minutes early (or log in 5 minutes early for virtual)",
    ],
    "donts": [
        "Don't speak negatively about previous employers or colleagues",
        "Don't exaggerate or fabricate experience — verify claims you make",
        "Don't interrupt the interviewer",
        "Don't ask about salary/benefits in the first round unless asked",
        "Don't give one-word answers — always elaborate with context and results",
        "Don't check your phone during the interview",
        "Don't show disinterest or appear unprepared",
        "Don't ask questions that are easily answered by the company website",
    ],
    "salary_negotiation": [
        "Always let the employer make the first salary offer if possible",
        "Counter with a range where your target is at the lower end",
        "Justify with market data (Levels.fyi, Glassdoor, LinkedIn Salary)",
        "Consider the full package: equity, bonus, PTO, remote, learning budget",
        "Get the offer in writing before resigning from current job",
        "It's always acceptable to ask for 24–48 hours to review an offer",
    ],
    "virtual_interview_tips": [
        "Test your internet connection, camera, and microphone beforehand",
        "Use a quiet, well-lit space with a neutral background",
        "Look at the camera, not the screen, when speaking",
        "Have water nearby and notes out of camera frame",
        "Close unnecessary browser tabs and notifications",
    ],
}

# ── Improvement Tips ─────────────────────────────────────────────────────────
IMPROVEMENT_TIPS = {
    "communication": [
        "Practice the Feynman Technique: explain concepts as if to a 10-year-old",
        "Record yourself answering interview questions and review the playback",
        "Join Toastmasters or a public speaking club",
        "Use the STAR framework for every behavioral answer",
    ],
    "technical": [
        "Solve 2–3 LeetCode problems daily (focus on medium-level)",
        "Read system design resources: Designing Data-Intensive Applications (Kleppmann)",
        "Contribute to open-source projects to demonstrate real-world code quality",
        "Write a technical blog post monthly to solidify knowledge",
    ],
    "confidence": [
        "Mock interviews with peers or on platforms like Pramp / interviewing.io",
        "Prepare and rehearse answers to the 20 most common questions",
        "Reframe nervousness as excitement — the physiological response is identical",
        "Use the 'power pose' technique 2 minutes before the interview",
    ],
    "research": [
        "Set up Google Alerts for the target company",
        "Follow the company's engineering blog and LinkedIn",
        "Review recent earnings calls or press releases for strategic context",
        "Connect with current employees on LinkedIn for informal insights",
    ],
}
