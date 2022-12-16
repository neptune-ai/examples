# Description

Include a summary of the changes and the related issue.

__Related to:__ `(ClickUp/JIRA task name)`

---

## ❔ This change

- [ ] adds a new feature
- [ ] fixes breaking code
- [ ] is cosmetic (refactoring/reformatting)

---

## ✔️ Pre-merge checklist

Mark which items have been completed:

- [ ] Refactored code ([sourcery](https://sourcery.ai/))
- [ ] Tested code locally
- [ ] Added code to GitHub tests ([notebooks](workflows/test-notebooks.yml), [scripts](workflows/test-scripts.yml))
- [ ] Updated GitHub [README](../README.md)

### 🪝 Run the following to both [pre-commit](https://pre-commit.com/) hooks and JupyterLab (turn on `format on save`)

- [ ] Stripped outputs from notebooks - ([nbstripout](https://pypi.org/project/nbstripout/))
- [ ] Sorted imports - ([isort](https://pycqa.github.io/isort/))
- [ ] Formatted code - ([back](https://github.com/psf/black))

---

## 🧪 Test Configuration

- OS:
- Python version:
- Neptune version:
- Affected libraries with version:
