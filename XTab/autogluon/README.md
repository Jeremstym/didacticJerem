# XTab: Cross-Table Pretrained Transformers

## Modifications
- Only kept submodules required to run the implementation of the FT-Transformer found in the _multimodal_ submodule; 
- Disabled dependencies that are not called when running the aforementioned FT-Transformer;
- Fix deprecated _lightning_ imports to bump dependency version from 1.7.X to 2.X;
- Removed dependency version constraints for remaining dependencies (except _nptyping_ which is limited to 1.X)
