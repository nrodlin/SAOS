# Interaction Matrix Helper

## IM and modal basis files

`save_IM` and `save_modalBasis` write the attribute `basis_convention` (`BASIS_CONVENTION`, currently 2: Noll sign of the
sine Zernikes and piston-free bases of `nValidAct - 1` modes, since 2026-10-27). Files without it are convention 1. An IM
is only valid with a modal basis of its own convention: `load_IM` and `load_modalBasis` raise `ValueError` when they do not
match (the sine Zernikes changed sign, so mixing them would invert the feedback of those modes). Load the modal basis file
saved with the IM, or measure the IM again.

`load_IM` puts every IM back in its own (DM, light path) entry: the groups `LP<i>/IM<j>` of the file are matched by index
first, so identical light paths or DMs of identical properties are not mixed up.

::: SAOS.InteractionMatrixHandler.InteractionMatrixHandler
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true