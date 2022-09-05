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

### 📄 Update documentation

Coordinate with @normandy7

- [ ] Updated content in the [docs](https://docs.neptune.ai)
- [ ] Added to all relevant docs index pages (e.g., [Examples](https://docs.neptune.ai/getting-started/examples), [Integrations intro](https://docs.neptune.ai/integrations-and-supported-tools/intro), [Model training index](https://docs.neptune.ai/integrations-and-supported-tools/model-training))
- [ ] Added to examples [README](../README.md)

---

## 🧪 Test Configuration

- OS:
- Python version:
- Neptune version:
- Affected libraries with version:

---

## ✔️ Post-merge checklist

These tasks should be done after the code is merged to main. Create a subtask under the main ClickUp task to have these on your radar and share the subtask here.

__Post-merge subtask:__ `(ClickUp/JIRA task name)`

- [ ] Update relevant blogs with impacted code/functionality (Contact @Patrycja-J for the list of blogs)
- [ ] Add functionality under the Neptune column of the `competitor comparison sheet` (Coordinate with @SiddhantSadangi)
- [ ] Update integration README (Contact @Patrycja-J)
