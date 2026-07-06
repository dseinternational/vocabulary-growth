# Contributing

**We welcome partners interested in developing and evaluating statistical models, evaluating and interpreting findings, and sharing original data.**

## Contributing data

We invite researchers and practitioners to share parent-reported vocabulary data relating to children with Down syndrome. Shared data may include MacArthur-Bates Communicative Development Inventory (MB-CDI) data, including all variations, translations and adaptations of the MB-CDI. They may also include data collected using the Vocabulary Checklists published by Down Syndrome Education International. They may also include any other form that presents a comprehensive list of early words against which a participant can indicate a child understands and/or says each word.

The data should include:

- **Instrument** (e.g. "MB-CDI Words and Gestures")
- **Subject identifier** (required if multiple observations of any child)
- **Age of child in months** (at the time the form was completed/updated)

One or more of the following total counts at the time the form was completed/updated:

- **Words understood** (whether spoken or signed or not)
- **Words spoken**
- **Words signed but not spoken**
- **Words signed and spoken**
- **Words signed whether or not spoken**

### Sharing data: privacy and consent

Please share only de-identified data. Before sending anything, ensure that:

- **No direct identifiers are included.** Replace any child or family name, address, NHS or medical-record number, and date of birth with non-identifying values. The subject identifier should be a pseudonymous code that you assign; keep any code-to-identity key yourself and do not share it with us.
- **Age, not date of birth.** Provide age in months at the time the form was completed, rather than a date of birth.
- **You have a lawful basis to share.** By contributing, you confirm you have the participants' consent and/or the ethical approval needed to share de-identified data for research and to have it published openly (see licensing below).

If you are unsure whether your data can be shared, contact us before sending it and we can talk it through.

### Licensing of contributed data

Data contributed to this study is published in this repository under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence, the same licence as the rest of the dataset (see [data/LICENSE](data/LICENSE)). By contributing data you confirm that you hold the rights necessary to share it under these terms.

You retain copyright in your data; CC BY 4.0 means others may reuse it provided they give appropriate credit. We will attribute the contributing study or researcher «how — e.g. in a data sources file and in published reports». If you would prefer a particular form of attribution, let us know.

### How to share data

Please get in touch via research@dseinternational.org before sending data, so we can agree a secure transfer method — we would rather not receive participant data as an unencrypted email attachment. We are happy to receive any tabular format (spreadsheets, statistical-package files or CSV), and we can undertake data preparation from your source files if helpful — for example, extracting columns or deriving age in months.

## Contributing code, models and reports

Code contributions are welcome by pull request and are accepted under the project's AGPL-3.0 licence (see [LICENSE](LICENSE)); documentation and reports are under CC BY 4.0 (see [docs/LICENSE](docs/LICENSE)).

### Commit messages

Please follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): a `<type>(optional scope): <summary>` subject line in the imperative mood, with any detail and rationale in the body. Common types are `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, and `chore`, and you can reference the issue a change closes (`Closes #123`) in the body or pull-request description. This keeps the project history readable and consistent.
