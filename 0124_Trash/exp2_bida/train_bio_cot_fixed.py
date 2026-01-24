def train_epoch(model, train_loader, criterion, optimizer, scaler, args, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ot_loss = 0.0
    total_consist_loss = 0.0
    total_adv_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n🔍 开始训练循环，总batch数: {len(train_loader)}")
    print(f"   数据加载器类型: {type(train_loader)}")
    print(f"   开始迭代数据加载器...")
    
    import time
    loop_start = time.time()
    
    try:
        pbar = tqdm(train_loader, desc='Training')
        print(f"   ✅ tqdm进度条创建成功")
        
        # 尝试获取第一个batch
        print(f"   ⏱️ 开始获取第一个batch...")
        iter_start = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 0:
                iter_time = time.time() - iter_start
                print(f"   ✅ 第一个batch获取成功，耗时: {iter_time:.2f}秒")
            
            # 添加调试信息（第一个batch）
            if batch_idx == 0:
                print(f"\n🔍 调试: 处理第一个batch...")
                print(f"   Batch类型: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"   Batch keys: {list(batch.keys())}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"   {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                        else:
                            print(f"   {k}: type={type(v)}")
                start_time = time.time()
                print(f"   ⏱️ 开始处理batch {batch_idx}...")
            
            # 解析batch
            if isinstance(batch, dict):
                if batch_idx == 0:
                    parse_start = time.time()
                    print(f"   ⏱️ 开始解析batch数据...")
                
                oct_feat = batch['oct_features'].to(device)
                colpo_feat = batch['colposcopy_features'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                center_labels = batch.get('center_id', torch.zeros(len(labels), dtype=torch.long)).to(device)
                
                # 从clinical_feat构建clinical_data
                batch_size = clinical_feat.size(0)
                clinical_data = {
                    'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size)],
                    'tct': [],
                    'age': [int(clinical_feat[i, 0].item() * 100) for i in range(batch_size)]
                }
                tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
                for i in range(batch_size):
                    tct_onehot = clinical_feat[i, 2:7]
                    tct_idx = tct_onehot.argmax().item()
                    clinical_data['tct'].append(tct_categories[tct_idx])
                
                if batch_idx == 0:
                    parse_time = time.time() - parse_start
                    print(f"   ✅ 数据解析完成，耗时: {parse_time:.2f}秒")
                    print(f"   ⏱️ 开始移动到GPU...")
                    move_start = time.time()
            else:
                if batch_idx == 0:
                    print(f"   ⚠️ Batch不是字典类型，跳过")
                continue
            
            # 移动到GPU
            if batch_idx == 0:
                if isinstance(batch, dict):
                    move_time = time.time() - move_start
                    print(f"   ✅ 数据移动到GPU完成，耗时: {move_time:.2f}秒")
                print(f"   ⏱️ 开始前向传播...")
                forward_start = time.time()
            
            optimizer.zero_grad()
            
            with autocast():
                if batch_idx == 0:
                    print(f"   ⏱️ 调用model.forward...")
                    model_start = time.time()
                
                outputs = model(
                    oct_features=oct_feat,
                    colpo_features=colpo_feat,
                    clinical_features=clinical_feat,
                    clinical_data=clinical_data,
                    center_labels=center_labels,
                    return_loss_components=True,
                    use_counterfactual=True
                )
                
                if batch_idx == 0:
                    model_time = time.time() - model_start
                    print(f"   ✅ 模型前向传播完成，耗时: {model_time:.2f}秒")
                    print(f"   ⏱️ 计算损失...")
                    loss_start = time.time()
                
                logits = outputs['logits']
                cls_loss = criterion(logits, labels)
                
                if batch_idx == 0:
                    loss_time = time.time() - loss_start
                    print(f"   ✅ 分类损失计算完成，耗时: {loss_time:.2f}秒")
                
                loss_components = outputs['loss_components']
                ot_loss = loss_components['L_ot']
                consist_loss = loss_components['L_consist']
                adv_loss = loss_components['L_adv']
                
                # 检查损失是否有nan或inf
                if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                    cls_loss = torch.tensor(0.0, device=cls_loss.device, requires_grad=True)
                if torch.isnan(ot_loss) or torch.isinf(ot_loss):
                    ot_loss = torch.tensor(0.0, device=ot_loss.device, requires_grad=True)
                if torch.isnan(consist_loss) or torch.isinf(consist_loss):
                    consist_loss = torch.tensor(0.0, device=consist_loss.device, requires_grad=True)
                if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                    adv_loss = torch.tensor(0.0, device=adv_loss.device, requires_grad=True)
                
                # 总损失
                total_loss_batch = (
                    args.lambda_cls * cls_loss +
                    args.lambda_ot * ot_loss +
                    args.lambda_consist * consist_loss +
                    args.lambda_adv * adv_loss
                )
                
                if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                    total_loss_batch = args.lambda_cls * cls_loss
            
            if batch_idx == 0:
                forward_time = time.time() - forward_start
                print(f"   ✅ 前向传播总耗时: {forward_time:.2f}秒")
                print(f"   ⏱️ 开始反向传播...")
                backward_start = time.time()
            
            # 检查损失是否有效
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                if batch_idx == 0:
                    print(f"   ⚠️ 损失无效，跳过batch")
                continue
            
            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx == 0:
                backward_time = time.time() - backward_start
                total_time = time.time() - start_time
                print(f"   ✅ 反向传播耗时: {backward_time:.2f}秒")
                print(f"   ✅ 总耗时: {total_time:.2f}秒")
                print(f"   ✅ 第一个batch完成！\n")
            
            # 统计
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_ot_loss += ot_loss.item()
            total_consist_loss += consist_loss.item()
            total_adv_loss += adv_loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Cls': f'{cls_loss.item():.4f}',
                'OT': f'{ot_loss.item():.4f}',
                'Consist': f'{consist_loss.item():.4f}',
                'Adv': f'{adv_loss.item():.4f}'
            })
    except Exception as e:
        print(f"\n❌ 训练循环异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'ot_loss': total_ot_loss / len(train_loader),
        'consist_loss': total_consist_loss / len(train_loader),
        'adv_loss': total_adv_loss / len(train_loader),
        'acc': 100. * correct / total
    }

