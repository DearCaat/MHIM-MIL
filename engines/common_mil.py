class CommonMIL():
	def __init__(self,args) -> None:
		self.training = True

	def init_func_train(self,args,**kwargs):
		self.training = True
	
	def init_func_val(self,args,**kwargs):
		self.training = False
	
	def after_get_data_func(self,args,**kwargs):
		pass

	def forward_func(self,args,model,model_ema,bag,label,criterion,batch_size,i,epoch,n_iter,pos,**kwargs):
		pad_ratio=0.
		kn_std = 0.

		if args.model == 'mhim':
			if model_ema is not None:
				cls_tea,attn = model_ema.forward_teacher(bag)
			else:
				attn,cls_tea = None,None

			cls_tea = None if args.aux_alpha == 0. else cls_tea

			if args.baseline == 'dsmil':
				logits, aux_loss,patch_num,keep_num = model(bag,attn,cls_tea[0],i=n_iter)
				logits = 0.5*logits[0].view(batch_size,-1)+0.5*logits[1].view(batch_size,-1)
			else:
				logits, aux_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=n_iter)

		elif args.model == 'mhim_pure':
			if args.baseline == 'dsmil':
				logits, aux_loss,patch_num,keep_num = model.pure(bag)
				logits = 0.5*logits[0].view(batch_size,-1)+0.5*logits[1].view(batch_size,-1)
			else:
				logits, aux_loss,patch_num,keep_num = model.pure(bag)
		elif args.model in ('clam_sb','clam_mb','dsmil'):
			logits, aux_loss, _ = model(bag,label=label,loss=criterion,pos=pos)
			keep_num = patch_num = bag.size(1)
		else:
			logits = model(bag,pos=pos)
			try:
				aux_loss, patch_num, keep_num = 0., bag.size(1), bag.size(1)
			except:
				aux_loss, patch_num, keep_num = 0., bag[0].size(0), bag[0].size(0)

		return logits,label,aux_loss,patch_num,keep_num,pad_ratio,kn_std
	
	def after_backward_func(self,args,**kwargs):
		pass
	
	def final_train_func(self,args,**kwargs):
		pass

	def validate_func(self,args,model,bag,label,criterion,batch_size,i,pos,epoch=None,**kwargs):
		if args.model in ('mhim','mhim_pure'):
			logits = model.forward_test(bag)
			if args.baseline == 'dsmil':
				logits = logits[0]
		elif args.model == 'dsmil':
			logits,_ = model(bag,pos=pos,epoch=epoch)
		else:
			logits = model(bag,pos=pos,epoch=epoch)

		if (args.model == 'mhim' and isinstance(logits,(list,tuple))) or (args.model == 'mhim_pure' and args.baseline == 'dsmil'):
			logits = 0.5*logits[0]+0.5*logits[1]
		
		return logits,label