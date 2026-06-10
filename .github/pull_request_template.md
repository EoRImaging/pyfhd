<!--- Provide a general summary of your changes in the Title above -->

## Description
<!--- Describe your changes in detail -->

## Motivation and Context
<!--- Why is this change required? What problem does it solve? -->
<!--- If it fixes an open issue, please link to the issue here. If this PR closes an issue, put the word 'closes' before the issue link to auto-close the issue when the PR is merged. -->

## Types of Changes
- [ ] Bug Fixes
- [ ] New Feature(s) or Translations
- [ ] New tests
- [ ] Breaking Changes
- [ ] Refactoring/Internal restructuring
- [ ] Documentation
- [ ] Version Change
- [ ] Build or CI Change
- [ ] Other

## Checklist
<!--- Please remove any checklists that don't apply to your change type(s)-->
<!--- Go over all the following points, and replace the space with an `x` in all the boxes that apply. -->
<!--- If you're unsure about any of these, don't hesitate to ask. We're here to help! -->
**New Features**
- [ ] If this is a new translation from the IDL FHD, I have read the Translation
      Contribution guide.
- [ ] I have added or updated the docstrings associated with my feature using
      the numpydoc docstring format.
- [ ] I have updated an existing tutorial or created a new tutorial to show how
      use the new feature.
- [ ] I have added new tests or changed existing tests to cover the new feature.
- [ ] I have updated the [Changelog](https://pyfhd.readthedocs.io/en/latest/changelog/changelog.html)
      with details on my new feature in the unreleased section (should be at the
      top of the relevant section -- changes are in reverse chronological order).

**New Test Data**
- [ ] My new test data is as small as practicable.
- [ ] I have added any new test data to a PR on the pyfhd-datasets repo.
- [ ] My PR on pyfhd-datasets repo has been merged and a new release has been done.
- [ ] I have updated the pooch information (dataset repo version, yaml, registry file)

**Existing Tests**
- [ ] If some tests fail and they are **meant** to, have they been changed?
      If so mention the exact tests that were changed and why here.
- [ ] Have the changes to existing tests been documented either through comments,
      changes to the docstring in ether the test or the associated functions?

**Breaking Changes**
- [ ] I have added new tests or changed existing tests to cover the breaking change.
- [ ] I have updated the [Changelog](https://pyfhd.readthedocs.io/en/latest/changelog/changelog.html)
      with details on the breaking changes in the unreleased section (should be at
      the top of the relevant section -- changes are in reverse chronological order).
- [ ] The breaking changes have been noted as a warning in any relevant tutorials
      in the documentation
- [ ] The breaking changes have been noted as a warning in any relevant functions
      in the API documentation, this should be put under the Warnings section in
      the numpydoc for the associated functions, please note the version that the
      breaking change is **not** in. For example if the current version is 1.1
      and you make the breaking change for it to be in version 1.2, say there has
      been a breaking change for versions <=1.1 and indicate what the change is.

**Refactoring/Internal restructuring**
- [ ] I have added or updated the docstrings for any new or updated funtions
      using the numpydoc docstring format.
- [ ] I am maintaining comparibility to FHD so other developers can tell where
      to find related functionality between pyfhd and FHD. If functions are
      structured or named differently from FHD, I have documented what the related
      FHD programs and/or functions are in docstrings.
- [ ] Existing tests pass and do not run slower than on main.

**Bug Fixes**
- [ ] I have linked all issues associated with the bugs above.
- [ ] I have added new tests or adjusted existing tests to cover bug (check the
Existing Tests Checklist above)
- [ ] I have updated the [Changelog](https://pyfhd.readthedocs.io/en/latest/changelog/changelog.html)
      with details on the bug fix in the unreleased section (should be at the top
      of the relevant section -- changes are in reverse chronological order).

**Documentation Updates**
- [ ] The documentation is able to build successfully with any new changes and
      they are visible in my own build

**Version**
- [ ] I have updated the changelog to put all the previous unreleased changelog
      into a new version
- [ ] I have noted dependency changes since the last version

**Build/CI**
- [ ] I have added a badge for any new CI actions or setups
- [ ] I have created issues for any expected future CI updates (e.g. undo pinning etc.)
