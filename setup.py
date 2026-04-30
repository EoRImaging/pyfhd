from setuptools import setup


# define the branch scheme. This is the only way to get the branch name into theversion
def branch_scheme(version):
    """
    Local version scheme that adds the branch name for absolute reproducibility.

    If and when this is added to setuptools_scm this function can be removed.
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch == "main":
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            version_str = version.format_choice(
                "+{node}.{branch}", "+{node}.{branch}.dirty"
            )
            version_str = version_str.replace("/", ".")
            return version_str


setup(
    use_scm_version={"local_scheme": branch_scheme},
)
