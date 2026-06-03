# Pull Request Title

General Description for your Pull Request goes here in terms of the main features it adds or bugfixes in the pull request.

Add any motivation or context here, especially if a particular issue sparked this pull request, make sure it gets tagged and mentioned with the proper keyword to close it in the changelog below!

Please also remove any changelog sections that don't match/relate to your pull request to ensure the changelog in the pull request is succinct, except for additional Translation, any additional translation will likely include all changes and checklists.

## Types of Changes
- [ ] Bug Fixes
- [ ] New Feature(s)
- [ ] Breaking Changes
- [ ] Documentation
- [ ] Version Change
- [ ] Translation from FHD
- [ ] Build or CI Change

## Changelog

### Breaking Changes!
* If any code you have produced breaks backwards compatibility with previous versions and will break existing codebases put those changes here

### New Features
* Add any new features here and if they were from an issue, ideally keep the issue open and have it get resolved through the pull request being merged and closed using the closing keywords followed by the issue number [Linking a pull request to an issue using a keyword](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword).

### Bug Fixes
* Any any bug fixes here and if they were from an issue, ideally keep the issue open and have it get resolved through the pull request being merged and closed using the closing keywords followed by the issue number [Linking a pull request to an issue using a keyword](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword).

### Test Changes
* Any new tests added mention them here
* Any changes to existing tests and the reasons for why they have been changed also noted.

### Dependency Changes
* Any new dependencies, add them here

### Version Changes
* Note what version you're getting ready for and the changelog changes, try to go through previous pull requests after the current release and try to add things to the changelog if they have been missed.

### Build/CI changes
* Note any new CI setups, changes of the build going to [PyPI](https://pypi.org/) and [Conda-Forge](https://conda-forge.org/)

### Translation Changes
* If you translated anything from the original FHD, please explain why

## Checklist

### General PR Checklist
  - [ ] I have the read the contribution guide
  - [ ] Add all the above to the [Changelog](https://pyfhd.readthedocs.io/en/latest/changelog/changelog.html) into the unreleased section
### New Features Checklist
  - [ ] New Features have all new functions documented using the numpydoc docstring format
  - [ ] Updated an existing tutorial or created a tutorial to use the new feature in the documentation
  - [ ] Have added new tests or changed existing tests to cover the new feature
### Existing Tests Checklist
  - [ ] If some tests fail and they are **meant** to, have they been changed? if so mention the exact tests that were changed and why here
  - [ ] Have the changes to existing tests been documented either through comments, changes to the docstring in ether the test or the associated functions?
### Breaking Changes Checklist
  - [ ] The breaking changes have been noted in the change log
  - [ ] The breaking changes have been noted as a warning in any relevant tutorials in the documentation
  - [ ] The breaking changes have been noted as a warning in any relevant functions in the API documentation, this should be put under the Warnings section in the numpydoc for the associated functions, please note the version that the breaking change is **not** in. For example if the current version is 1.1 and you make the breaking change for it to be in version 1.2, say there has been a breaking change for versions <=1.1 and indicate what the change is.
### Bug Fixes Checklist
  - [ ] Added all bug fixes with any issues associated with them linked.
  - [ ] Added new tests to cover bugs
  - [ ] Adjusted existing tests to cover bug, in which case check the [Existing Tests Checklist](#existing-tests-checklist)
### Documentation Checklist
  - [ ] The documentation is able to build successfully with any new changes and they are visible in your own build
### Version Checklist
  - [ ] Updated the changelog to put all the previous unreleased changelog into a version
  - [ ] Noted dependency changes since the last version
### Build/CI
  - [ ] Added a badge for any new CI actions or setups
  - [ ] Existing badges remain unaffected, if not, document why in changelog
### Translation Checklist
  - [ ] I have read the Translation Contribution guide
  - [ ] I have created new tests and added them to the FHD Test source for any other developers to also test
  - [ ] I have followed all previous checklists as additional Translation of FHD usually includes all of the above
