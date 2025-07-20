<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">PyExner</h1></p>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/jkhansell/PyExner?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/jkhansell/PyExner?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/jkhansell/PyExner?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/jkhansell/PyExner?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ REPLACE-ME</code>

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Project Structure

```sh
└── PyExner/
    ├── LICENSE
    ├── PyExner
    │   ├── __init__.py
    │   ├── config.py
    │   ├── domain
    │   ├── integrators
    │   ├── io
    │   ├── runtime
    │   ├── solvers
    │   ├── state
    │   └── utils
    ├── PyExner.png
    ├── README.md
    ├── pyproject.toml
    └── tests
        ├── domain
        └── runtime
```


###  Project Index
<details open>
	<summary><b><code>PYEXNER/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/pyproject.toml'>pyproject.toml</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- PyExner Submodule -->
		<summary><b>PyExner</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/config.py'>config.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
			<details>
				<summary><b>runtime</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/runtime/driver.py'>driver.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>solvers</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/base.py'>base.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/roe_exner_solver.py'>roe_exner_solver.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/registry.py'>registry.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/roe_solver.py'>roe_solver.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
					<details>
						<summary><b>kernels</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/kernels/hllc.py'>hllc.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/kernels/roe_exner.py'>roe_exner.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/solvers/kernels/roe.py'>roe.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>io</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/io/visualizer.py'>visualizer.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/io/diagnostics.py'>diagnostics.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>integrators</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/integrators/base.py'>base.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/integrators/forwardeuler.py'>forwardeuler.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/integrators/registry.py'>registry.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/integrators/RK2.py'>RK2.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>utils</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/utils/utils.py'>utils.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/utils/constants.py'>constants.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>domain</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/domain/mesh.py'>mesh.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/domain/boundary_registry.py'>boundary_registry.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/domain/halo_exchange.py'>halo_exchange.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
					<details>
						<summary><b>boundaries</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/domain/boundaries/roe_reflective.py'>roe_reflective.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/domain/boundaries/roe_transmissive.py'>roe_transmissive.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>state</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/state/registry.py'>registry.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/state/roe_exner_state.py'>roe_exner_state.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/jkhansell/PyExner/blob/master/PyExner/state/roe_state.py'>roe_state.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with PyExner, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install PyExner using one of the following methods:

**Build from source:**

1. Clone the PyExner repository:
```sh
❯ git clone https://github.com/jkhansell/PyExner
```

2. Navigate to the project directory:
```sh
❯ cd PyExner
```

3. Install the project dependencies:

echo 'INSERT-INSTALL-COMMAND-HERE'



###  Usage
Run PyExner using the following command:
echo 'INSERT-RUN-COMMAND-HERE'

###  Testing
Run the test suite using the following command:
echo 'INSERT-TEST-COMMAND-HERE'

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/jkhansell/PyExner/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/jkhansell/PyExner/issues)**: Submit bugs found or log feature requests for the `PyExner` project.
- **💡 [Submit Pull Requests](https://github.com/jkhansell/PyExner/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/jkhansell/PyExner
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/jkhansell/PyExner/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=jkhansell/PyExner">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---