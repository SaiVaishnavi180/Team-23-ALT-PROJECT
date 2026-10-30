import json, random, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUT = "out"
import os; os.makedirs(OUT, exist_ok=True)

ROLES = {
 "Data Scientist": ["python","statistics","machine learning","pandas","numpy","scikit-learn","data visualization","sql","deep learning","feature engineering","hypothesis testing"],
 "ML Engineer": ["python","machine learning","deep learning","tensorflow","pytorch","docker","mlops","model deployment","sql","git","kubernetes"],
 "Web Developer": ["html","css","javascript","react","node.js","rest api","sql","git","mongodb","typescript"],
 "Data Analyst": ["sql","excel","power bi","tableau","data visualization","statistics","python","data cleaning","reporting"],
 "DevOps Engineer": ["docker","kubernetes","ci/cd","jenkins","linux","terraform","aws","git","ansible","monitoring","bash"],
 "Cybersecurity Analyst": ["network security","penetration testing","siem","linux","incident response","cryptography","firewalls","vulnerability assessment","python"],
 "Android Developer": ["kotlin","java","android sdk","rest api","git","sqlite","firebase","jetpack compose","ui design"],
 "Cloud Engineer": ["aws","azure","terraform","linux","networking","docker","kubernetes","python","iam","serverless"],
}
VARIANTS = {"machine learning":["machine learning","ML"],"javascript":["JavaScript","JS"],
 "scikit-learn":["scikit-learn","sklearn"],"power bi":["Power BI","PowerBI"],"node.js":["Node.js","NodeJS"],
 "ci/cd":["CI/CD","continuous integration"],"kubernetes":["Kubernetes","K8s"],"deep learning":["deep learning","neural networks"],
 "rest api":["REST API","RESTful services"],"data visualization":["data visualization","data visualisation"],
 "mlops":["MLOps","ML operations"],"siem":["SIEM","security information and event management"],"aws":["AWS","Amazon Web Services"]}
ALL_SKILLS = sorted({s for v in ROLES.values() for s in v})
def variant(s): return random.choice(VARIANTS.get(s, [s]))
FILLER = ["We are looking for a motivated candidate to join our growing team.",
 "The role involves close collaboration with cross-functional stakeholders.",
 "Strong communication and problem-solving abilities are expected.",
 "You will own features end to end in an agile environment.",
 "Prior experience in a product company is a plus.",
 "Candidates should be comfortable working with ambiguous requirements."]
STARTS = ["Requirements include", "You should be proficient in", "Hands-on experience with", "Key skills:", "Must know"]

# ---------- 1. synthetic job-description dataset ----------
rows = []
for role, core in ROLES.items():
    others = [s for r, v in ROLES.items() if r != role for s in v if s not in core]
    for _ in range(150):
        k = random.randint(5, min(8, len(core)))
        pick = random.sample(core, k) + random.sample(others, random.randint(1, 3))
        random.shuffle(pick)
        txt = f"{random.choice(STARTS)} " + ", ".join(variant(s) for s in pick) + ". " + " ".join(random.sample(FILLER, 3))
        rows.append((txt, role))
jobs = pd.DataFrame(rows, columns=["text", "role"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
jobs.to_csv(f"{OUT}/jobs_demo.csv", index=False)

Xtr, Xte, ytr, yte = train_test_split(jobs.text, jobs.role, test_size=0.2, stratify=jobs.role, random_state=SEED)

# ---------- 2. role classification: model comparison ----------
vec = lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2)
models = {
 "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
 "Logistic Regression": LogisticRegression(C=10, max_iter=1000, random_state=SEED),
 "Linear SVM": LinearSVC(C=1.0, random_state=SEED),
 "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
res = {}
fitted = {}
for name, m in models.items():
    pipe = Pipeline([("tfidf", vec()), ("clf", m)])
    cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro")
    pipe.fit(Xtr, ytr); pr = pipe.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pr, average="macro", zero_division=0)
    res[name] = dict(cv_f1=float(cvs.mean()), cv_std=float(cvs.std()), acc=float(accuracy_score(yte, pr)),
                     prec=float(p), rec=float(r), f1=float(f))
    fitted[name] = (pipe, pr)
best = max(res, key=lambda n: res[n]["cv_f1"])
bp, bpred = fitted[best]
labels = sorted(ROLES)
cm = confusion_matrix(yte, bpred, labels=labels)
rep = classification_report(yte, bpred, labels=labels, output_dict=True, zero_division=0)

# ---------- 3. skill lexicon, extraction and gap analysis ----------
canon = {}
for s in ALL_SKILLS:
    for v in set([s] + VARIANTS.get(s, [])):
        canon[v.lower()] = s
def exact_extract(text):
    t = text.lower(); found = set()
    for v, s in canon.items():
        if re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", t): found.add(s)
    return found
cv_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
keys = list(canon)
KM = cv_char.fit_transform(keys)
def fuzzy_extract(text, thr=0.62):
    found = exact_extract(text)
    for ph in re.split(r"[,.;:]", text):
        ph = ph.strip().lower()
        if not ph or len(ph) > 40: continue
        sims = cosine_similarity(cv_char.transform([ph]), KM)[0]
        j = sims.argmax()
        if sims[j] >= thr: found.add(canon[keys[j]])
    return found

# role skill profiles learned from TRAINING postings only (required = appears in >= 40% of postings)
prof = {}
train_df = pd.DataFrame({"text": Xtr, "role": ytr})
for role in labels:
    docs = train_df[train_df.role == role].text.tolist()
    cnt = {s: 0 for s in ALL_SKILLS}
    for d in docs:
        for s in exact_extract(d): cnt[s] += 1
    prof[role] = {s: c / len(docs) for s, c in cnt.items() if c / len(docs) >= 0.40}

def noisy(s):
    t = random.random()
    base = variant(s)
    if t < 0.4 and len(base) > 4:
        i = random.randint(1, len(base) - 2); return base[:i] + base[i + 1:]       # dropped letter
    if t < 0.7: return base.replace(" ", "").replace("-", "")                         # spacing variants
    return base + random.choice([" basics", " tools"])                               # extra word
students = []
for i in range(200):
    role = random.choice(labels)
    req = list(prof[role])
    known = [s for s in req if random.random() < 0.45]
    extra = random.sample([s for s in ALL_SKILLS if s not in req], 3)
    mentions = []
    for s in known + extra:
        mentions.append(noisy(s) if random.random() < 0.30 else variant(s))
    random.shuffle(mentions)
    text = "Skills and tools: " + ", ".join(mentions) + ". Interests: " + random.choice(["AI","open source","startups","teaching"]) + "."
    gap = set(req) - set(known)
    students.append(dict(role=role, text=text, known=set(known), gap=gap, req=set(req)))
def prf(pred_fn):
    tp = fp = fn = 0
    for st in students:
        found = pred_fn(st["text"])
        gap = st["req"] - found
        tp += len(gap & st["gap"]); fp += len(gap - st["gap"]); fn += len(st["gap"] - gap)
    p = tp / (tp + fp); r = tp / (tp + fn); return p, r, 2 * p * r / (p + r)
ex = prf(exact_extract); fz = prf(fuzzy_extract)
# extraction-level (known skills recovered)
def ext_rec(fn):
    tp = fp = fn_ = 0
    for st in students:
        f = fn(st["text"]); kn = st["known"]
        tp += len(f & kn); fp += len((f & st["req"]) - kn); fn_ += len(kn - f)
    return tp / (tp + fn_)
ext_exact, ext_fuzzy = ext_rec(exact_extract), ext_rec(fuzzy_extract)
readiness = float(np.mean([len(st["known"]) / len(st["req"]) for st in students]))
avg_gap = float(np.mean([len(st["gap"]) for st in students]))

# ---------- 4. course retrieval and ranking ----------
LEVELS = [("Introduction to {s}", "Beginner", "A beginner friendly course covering fundamentals and basic concepts of {s} with guided exercises."),
          ("{s} for Practitioners", "Intermediate", "Build real world projects and apply {s} to practical problems with industry examples."),
          ("Advanced {s} in Production", "Advanced", "Master advanced patterns, optimisation and best practices of {s} for professional use.")]
courses = []
for s in ALL_SKILLS:
    for t, lv, d in LEVELS:
        courses.append(dict(skill=s, level=lv, title=t.format(s=s.title()), desc=d.format(s=s)))
cdf = pd.DataFrame(courses); cdf.to_csv(f"{OUT}/courses_demo.csv", index=False)
cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
CM = cvec.fit_transform((cdf.title + " " + cdf.desc).tolist())
def recommend(skill_text, k=3):
    sims = cosine_similarity(cvec.transform([skill_text]), CM)[0]
    return np.argsort(-sims)[:k]
h1 = h3 = n = 0
for st in students:
    for s in st["gap"]:
        q = "learn " + noisy(s) if random.random() < 0.3 else "learn " + variant(s)
        idx = recommend(q, 3); n += 1
        h1 += cdf.skill[idx[0]] == s; h3 += any(cdf.skill[i] == s for i in idx)
hit1, hit3 = h1 / n, h3 / n

# worked example
ex_st = next(st for st in students if st["role"] == "Data Scientist" and len(st["gap"]) >= 4)
found = fuzzy_extract(ex_st["text"]); gap = [s for s in prof["Data Scientist"] if s not in found]
gap = sorted(gap, key=lambda s: -prof["Data Scientist"][s])
path = []
for s in gap:
    i = recommend("learn " + s, 1)[0]
    path.append(dict(skill=s, weight=round(prof["Data Scientist"][s], 2), course=cdf.title[i], level=cdf.level[i]))
example = dict(text=ex_st["text"], role="Data Scientist", found=sorted(found & set(prof["Data Scientist"])), gap=gap, path=path)

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(5.2, 3.0))
vc = jobs.role.value_counts().reindex(labels)
ax.barh(labels, vc.values, color="#2F5597"); ax.invert_yaxis(); ax.set_xlabel("Number of job descriptions")
for i, v in enumerate(vc.values): ax.text(v + 2, i, str(v), va="center")
ax.set_xlim(0, 180); plt.tight_layout(); plt.savefig(f"{OUT}/fig_dist.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted role"); ax.set_ylabel("Actual role"); plt.colorbar(im, fraction=0.04)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_cm.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.0))
names = list(res); x = np.arange(len(names)); w = 0.38
ax.bar(x - w / 2, [res[n]["acc"] for n in names], w, label="Accuracy", color="#2F5597")
ax.bar(x + w / 2, [res[n]["f1"] for n in names], w, label="Macro F1", color="#C55A11")
ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names]); ax.set_ylim(0.6, 1.02)
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_models.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 2.8))
lab = ["Precision", "Recall", "F1"]; x = np.arange(3)
ax.bar(x - w / 2, ex, w, label="Exact lexicon match", color="#7F7F7F")
ax.bar(x + w / 2, fz, w, label="Lexicon + char n-gram match", color="#2F5597")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0.5, 1.05); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_gap.png", dpi=200); plt.close()

per_class = {l: {k: float(rep[l][k]) for k in ("precision", "recall", "f1-score", "support")} for l in labels}
out = dict(n_jobs=len(jobs), n_train=len(Xtr), n_test=len(Xte), dist=vc.to_dict(), models=res, best=best, per_class=per_class,
           gap=dict(exact=ex, fuzzy=fz), extraction=dict(exact=ext_exact, fuzzy=ext_fuzzy),
           readiness=readiness, avg_gap=avg_gap, retrieval=dict(hit1=hit1, hit3=hit3, n=n),
           n_students=len(students), n_courses=len(cdf), n_skills=len(ALL_SKILLS),
           profiles={r: sorted(p, key=lambda s: -p[s]) for r, p in prof.items()}, example=example)
json.dump(out, open(f"{OUT}/results.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["models", "best", "gap", "extraction", "retrieval", "readiness", "avg_gap"]}, indent=1))
print(example)
