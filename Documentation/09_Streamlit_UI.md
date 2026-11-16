# 09 - Streamlit UI

**Owner:** Muhammad Usman  |  **Files:** `app/streamlit_app.py`, `app/assets/sepsis.png`, `.streamlit/config.toml`  |  **Run:** `streamlit run app/streamlit_app.py`

## Predict only on button press
The prediction code sits inside `if submitted:` where `submitted` is the result of the **Predict sepsis risk** button.
Changing a value does nothing until the button is pressed. The result is kept in `st.session_state["result"]`.
If the user changes a value or the alert mode after predicting, a yellow notice says the result is out of date.
Example buttons clear the old result, so the user must press Predict again.

## Layout
| Part | How it is built |
|---|---|
| Glass look | CSS: translucent white, `backdrop-filter: blur`, soft border and shadow, gradient background |
| Cards | `st.container(key="hero" / "inputs" / "result")` gives CSS class `st-key-...` |
| Header | Transparent PNG logo, gradient title, chips (model name, ROC-AUC, recall in both modes) |
| Inputs | 12 `st.number_input` boxes with a slider under each: 6 vitals, 4 labs, age, ICU hours. Sliders are coloured green / amber / red |
| Examples | Healthy and Sepsis-like buttons (`on_click` sets `st.session_state`) |
| Alert mode | Radio: **Balanced** or **High sensitivity** (changes the alert threshold, see below) |
| Result | Plotly gauge, risk badge, message box, warning-sign chips |
| What-if chart | Each abnormal value is set back to normal, one at a time, and the score drop is shown |
| History | Table of this session's predictions with CSV download |
| Tab 2 | All-model comparison table, high-sensitivity table, bar chart, saved settings, feature importance |

## Alert modes
| Mode | Threshold | Use |
|---|---|---|
| Balanced | F2-optimal (about 0.67) | Normal use. Fewer false alarms |
| High sensitivity | Keeps about 70% recall (about 0.39) | When missing a case is unacceptable. Many more false alarms |

## Risk levels
| Level | Rule |
|---|---|
| High | score at or above the threshold (alert) |
| Elevated | between 80% of the threshold and the threshold |
| Low | below that |

## Other details
- `@st.cache_resource` loads the pickle once.
- Warning-sign rules: HR over 90, Resp 22 or more, Temp over 38 or under 36, SBP 100 or less, MAP under 65, O2 under 92, Lactate over 2, WBC over 12 or under 4, Creatinine over 1.2, Platelets under 150.
- The score is a model output, not an exact probability (class weights change it).
- The app scores **one time point**. It does not see the patient's past hours.
