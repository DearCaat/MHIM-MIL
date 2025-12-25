import torch
from torch_geometric.data import Data, Batch


class BatchWSI(Batch):
    def __init__(self, batch=None, ptr=None, **kwargs):
        super().__init__(batch=batch, ptr=ptr, **kwargs)

    @classmethod  
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[], update_cat_dims={}):
        """
        Batch creation method adapted for PyTorch Geometric 2.x
        Special handling for edge_latent field to concatenate on dimension 0 to maintain consistency with old version
        
        Args:
            data_list: List of data objects
            follow_batch: Fields that need to track batch information
            exclude_keys: Fields to exclude
            update_cat_dims: Custom concatenation dimension field mapping
            
        Returns:
            BatchWSI: Batched data object
        """
        if not data_list:
            raise ValueError("data_list cannot be empty")

        default_special_fields = {'edge_latent': 0}
        
        special_cat_dims = {**default_special_fields, **update_cat_dims}
        
        temp_data_list = []
        special_items = {}
        
        for i, data in enumerate(data_list):
            data_copy = data.clone() if hasattr(data, 'clone') else Data(**{k: v for k, v in data})
            
            for key in special_cat_dims:
                if hasattr(data_copy, key):
                    if key not in special_items:
                        special_items[key] = []
                    special_items[key].append(getattr(data_copy, key))
                    delattr(data_copy, key)
            
            temp_data_list.append(data_copy)

        batch = super(BatchWSI, cls).from_data_list(
            temp_data_list,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys
        )
        
        for key, items in special_items.items():
            cat_dim = special_cat_dims[key]
            try:
                if all(isinstance(item, torch.Tensor) for item in items):
                    shapes = [list(item.shape) for item in items]
                    all_same_shape = len(set([tuple(s) for s in shapes])) == 1
                    
                    if key == 'edge_latent' and cat_dim == 0:
                        batch[key] = torch.cat(items, dim=0)
                        continue
                    
                    if (cat_dim == 0 and all_same_shape and len(items) > 1 and key != 'edge_latent'):
                        batch[key] = torch.stack(items, dim=0)
                        continue
                    
                    compatible = True
                    if len(shapes) > 1 and len(shapes[0]) > 0:
                        ref_shape = shapes[0].copy()
                        if 0 <= cat_dim < len(ref_shape):
                            ref_shape.pop(cat_dim)
                        
                        for shape in shapes[1:]:
                            check_shape = shape.copy()
                            if 0 <= cat_dim < len(check_shape):
                                check_shape.pop(cat_dim)
                            if check_shape != ref_shape:
                                compatible = False
                                break
                    
                    if compatible and len(items) > 0:
                        batch[key] = torch.cat(items, dim=cat_dim)
                    else:
                        if all_same_shape:
                            batch[key] = torch.stack(items, dim=0)
                        else:
                            shape_info = [f"item{i}: {tuple(s)}" for i, s in enumerate(shapes)]
                            batch[key] = items
                else:
                    batch[key] = items
                    
            except Exception as e:
                batch[key] = items

        batch.__class__ = cls
        return batch.contiguous()

