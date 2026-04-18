# cv-self-drive

Run `python self_drive_occ_grid.py <turn_type>` to run the corresponding turn code.
Where `<turn_type>` is either "left" or "right".

Run `python functional_tests_occ_grid.py <function_type>` to run the corresponding functional test.
Where `<function_type>` is:
* `right` for right turn
* `left` for left turn
* `pedlanechange` for pedestrian lane changing
* `curvedlanekeep` for curved lane keeping

`cv-depth-segmentation` has been added as a submodule. To pull updates from the submodule, run `git submodule update --init --recursive`.