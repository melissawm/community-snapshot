# community-snapshot
Tools for gathering metrics around open-source communities.

This repo constains a [JupyterBook](https://jupyterbook.org/) which takes information from a GitHub repository and creates summaries, plots and reports based on a few common metrics used to measure open source community health.

## GitHub access token

To run this book locally and apply the reports to your own repo, you need to [create a personal access token from GitHub](https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) and choose which repositories you want to analyze in `report.md`. 

## Building the book

To build this book, use

```
$ jupyter-book build community-snapshot/
```

---

This project is heavily based on the [jupyter-activity-snapshot](https://github.com/choldgraf/jupyter-activity-snapshot).
