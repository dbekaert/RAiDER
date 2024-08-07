# Contributing Guidelines #

This document is inspired by similar instructions from MintPY, ISCE, gdal and jupyterhub. 
These are several ways to contribute to the RAiDER framework:

* Submitting bug reports and feature requests in RAiDER
* Writing tutorials or jupyter-notebooks in RAiDER-docs
* Fixing typos, code and improving documentation
* Writing code for everyone to use

If you get stuck at any point you can create an [issue on GitHub](https://github.com/dbekaert/RAiDER/issues).

For more information on contributing to open source projects, [GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point if you are new to version control.

## Optional Dependencies

In order to better support the NISAR SDS (see: [#533](https://github.com/dbekaert/RAiDER/issues/533)), RAiDER has some optional dependencies:

* ISCE3
* Pandas
* Rasterio
* Progressbar

RAiDER distributes two conda packages, `raider-base` a lighter-weight package that does depend on the optional dependencies, and `raider` which includes all dependencies. When using, or adding new, optional dependenices in RAiDER, please follow this pattern:
1. When you import the optional dependency, handle import errors like:
   ```python
   try:
    import optional_dependency
   except ImportError:
    optional_dependency = None
   ```
   Note: you *do not* need to delay imports until use with this pattern.
2. At the top of any function/method that uses the optional dependency, throw if it's missing like:
   ```python
   if optional_dependency is None:
       raise ImportError('optional_dependency is required for this function. Use conda to install optional_dependency')
   ```
3. If you want to add type hints for objects in the optional_dependency, use a forward declaration like:
   ```python
   def function_that_uses_optional_dependency(obj: 'optional_dependency.obj'):
   ```
   Note: the typehint is a string here.

## Git workflows ##

### Setting up the development environment ###

Fork RAiDER from GitHub UI, and then

```
git clone https://github.com/dbekaert/RAiDER.git
cd RAiDER
git remote add my_user_name https://github.com/my_user_name/RAiDER.git
```

### Setting up the documentation environment ###

Fork RAiDER-docs from GitHub UI, and then

```
git clone https://github.com/dbekaert/RAiDER-docs.git
cd RAiDER-docs
git remote add my_user_name https://github.com/my_user_name/RAiDER-docs.git
```


### Updating your local master against upstream master ###

```
git checkout master
git fetch origin
# Be careful: this will loose all local changes you might have done now
git reset --hard origin/master
```

### Working with a feature branch ###

[Here](https://thoughtbot.com/blog/git-interactive-rebase-squash-amend-rewriting-history) is a great tutorial if you are new to rewriting history with git.

```
git checkout master
(potentially update your local master against upstream, as described above)
git checkout -b my_new_feature_branch

# do work. For example:
git add my_new_file
git add my_modifid_message
git rm old_file
git commit -a 

# you may need to resynchronize against master if you need some bugfix
# or new capability that has been added to master since you created your
# branch
git fetch origin
git rebase origin/master

# At end of your work, make sure history is reasonable by folding non
# significant commits into a consistent set
git rebase -i master (use 'fixup' for example to merge several commits together,
and 'reword' to modify commit messages)

# or alternatively, in case there is a big number of commits and marking
# all them as 'fixup' is tedious
git fetch origin
git rebase origin/master
git reset --soft origin/master
git commit -a -m "Put here the synthetic commit message"

# push your branch
git push my_user_name my_new_feature_branch
```

### Formatting and linting with [Ruff](https://docs.astral.sh/ruff/) ###

Format your code to follow the style of the project with:
```
ruff format
```
and check for linting problems with:
```
ruff check
```
Please ensure that any linting problems in your changes are resolved before
submitting a pull request.
> [!TIP]
> vscode users can [install the ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to run the linter automatically in the
editor.

### Issue a pull request from GitHub UI ###
commit locally and push. To get a reasonable history, you may need to

```
git rebase -i master
```

, in which case you will have to force-push your branch with 

```
git push -f origin my_new_feature_branch
```

Once a pull request is issued it will be reviewed by multiple members before it will be approved and integrated into the main.

### Things you should NOT do
(For anyone with push rights to RAiDER or RAiDER-docs) Never modify a commit or the history of anything that has been committed to https://github.com/dbekaert/RAiDER and https://github.com/dbekaert/RAiDER-docs.
