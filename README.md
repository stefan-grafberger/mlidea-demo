`mlidea` demo
===


This web app was created to demonstrate the functionality of `mlidea`: <https://github.com/stefan-grafberger/mlidea>

`mlidea` is a tool for Interactively Improving ML Code. It uses the [`mlinspect`](https://github.com/stefan-grafberger/mlinspect) project as a foundation, mainly for its plan extraction from native ML pipelines.

This demo app is built using [Streamlit](https://streamlit.io).

Requirements
---

Python 3.11

Usage
---

```shell
# Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip

# Install dependencies
SETUPTOOLS_USE_DISTUTILS=stdlib pip install -r requirements.txt

# Run the web app
streamlit run app.py --theme.base light
```

Visit <http://localhost:8501> in your browser.

<!-- TODO: Caching -->
<!-- TODO: Pages -->
<!-- TODO: Docker -->
<!-- TODO: Deployment -->

Notes 
---

* For debugging the app in PyCharm, see [here](https://stackoverflow.com/a/60172283)

License
---

This library is licensed under the Apache 2.0 License.
