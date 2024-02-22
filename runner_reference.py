def run_train_epoch(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        optimizer,
        scheduler):
    

    dl_dict, split_id_dict = demo_dataset.get_dataloaders(run_d)

    train_loader = dl_d['train']
    val_loader = dl_d['primary']['val']

    # stuff related to device
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]
    # set up loss func
    loss_func = get_loss_func(
        run_d["loss"],
        data_d["mz_bin_res"],
        agg=run_d["batch_loss_agg"])
    b_losses = []
    if run_d["lda_topic_loss"]:
        lda_loss_func = get_loss_func(
            "forw_kl",
            100,
            agg=run_d["batch_loss_agg"])
    # set up scaler
    scaler = get_scaler(run_d["amp"])
    # train
    model.train()
    # get embed dim for flag
    if run_d["flag"]:
        if isinstance(model, th.nn.DataParallel):
            embedders = model.module.embedders
        else:
            embedders = model.embedders
        gfv2_idx = [isinstance(embedder, GFv2Embedder)
                    for embedder in embedders].index(True)
        embed_dim = embedders[gfv2_idx].args.encoder_embed_dim
    # iterate
    for b_idx, b in get_pbar(
        enumerate(
            dl_d["train"]), run_d, desc="> train", total=len(
            dl_d["train"])):
        optimizer.zero_grad()
        b = data_to_device(b, dev, nb)
        b_output = model(
            data=b, 
            amp=run_d["amp"], 
            return_lda_pred=run_d["lda_topic_loss"])
        b_pred = b_output["pred"]
        b_targ = b["spec"]
        if run_d["flag"]:
            def forward(perturb):
                b_pred = model(data=b, perturb=perturb, amp=run_d["amp"])["pred"]
                return b_pred
            def backward(loss):
                # this backward is only meant for generating perturbations
                scaler.scale(loss).backward()
            n_graph, n_node = b["gf_v2_data"]["x"].shape[:2]
            b_perturb_shape = (n_graph, n_node, embed_dim)
            b_loss_agg, b_pred = flag_bounded(
                (model, forward, backward),
                b_perturb_shape,
                b_targ,
                optimizer,
                dev,
                loss_func,
                scaler,
                m=run_d["flag_m"],
                step_size=run_d["flag_step_size"],
                mag=run_d["flag_mag"],
                mask=None
            )
        else:
            b_loss_agg = loss_func(b_pred, b_targ)
        if run_d["lda_topic_loss"]:
            b_lda_pred = b_output["lda_pred"]
            b_lda_targ = b["lda_topic"]
            b_lda_loss_agg = lda_loss_func(b_lda_pred, b_lda_targ)
            b_loss_agg = b_loss_agg + run_d["lda_topic_loss_weight"] * b_lda_loss_agg
        # backpropagate loss
        scaler.scale(b_loss_agg / run_d["grad_acc_interval"]).backward()
        # take a gradient step if finished accumulating
        if step % run_d["grad_acc_interval"] == 0:
            # unscale then gradient clip
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(
                model.parameters(), run_d["clip_grad_norm"])
            scaler.step(optimizer)  # calls optimizer.step()
            scaler.update()
            if run_d["scheduler"] == "polynomial":
                # polynomial updates per-step
                scheduler.step()
        # increment step counter
        step += 1
        # update losses
        b_losses.append(b_loss_agg.detach().to("cpu").item())
    optimizer.zero_grad()
    train_spec_loss = np.mean(b_losses)
    if use_wandb:
        log_d = {
            "train_spec_loss_obj_mean": train_spec_loss,
            "epoch": epoch,
            "Epoch": epoch,
        }
        wandb.log(log_d, commit=False)
    return step, epoch, {}


def compute_metric_tables(
        pred,
        targ,
        mol_id,
        group_id,
        prefix,
        data_d,
        run_d,
        auxiliary=False,
        merge_group=False,
        compute_agg=False,
        compute_hist=False,
        groupby_mol=False,
        um_batch_size=10000,
        m_batch_size=1000):

    def merge_group_func(_pred, _targ, _group_id, _mol_id, _transform):
        # merge spectra by mol
        assert group_id is not None and mol_id is not None
        # translate tranform
        if _transform == "obj":
            t = data_d["transform"]
            def pp(x): return x
            n = data_d["spectrum_normalization"]
        elif _transform == "std":
            t = "none"
            def pp(x): return x
            n = "l1"
        elif _transform == "log":
            t = "log10over3"
            def pp(x): return x
            n = "l1"
        elif _transform == "cfm":
            t = "none"
            def pp(x): return cfm_postprocess(x, "l1")
            n = "l1"
        else:
            raise ValueError
        m_pred, m_group_id, m_mol_id = merge_spec(
            _pred, _group_id, t, n, _mol_id)
        m_pred = pp(m_pred)
        m_targ, _ = merge_spec(_targ, _group_id, t, n)
        return m_pred, m_targ, m_mol_id, m_group_id
    # batching
    um_num_batches = len(pred) // um_batch_size + int(len(pred) % um_batch_size != 0)
    # functions
    obj_sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
    obj_loss_func = get_loss_func(run_d["loss"], data_d["mz_bin_res"])
    cos_sim_func = get_sim_func("cos", data_d["mz_bin_res"])
    # do unmerged first
    sim_obj, loss_obj, sim_cos_std = [], [], []
    for b in get_pbar(range(um_num_batches),run_d,desc="> unmerged metrics"):
        b_pred = pred[b*um_batch_size:(b+1)*um_batch_size]
        b_targ = targ[b*um_batch_size:(b+1)*um_batch_size]
        # basic loss and sim
        b_sim_obj = obj_sim_func(b_pred, b_targ)
        b_loss_obj = obj_loss_func(b_pred, b_targ)
        sim_obj.append(b_sim_obj)
        loss_obj.append(b_loss_obj)
        if auxiliary:
            # just doing cos, forget about the other ones
            b_pred = process_spec(unprocess_spec(b_pred, data_d["transform"]),"none","l2")
            b_targ = process_spec(unprocess_spec(b_targ, data_d["transform"]),"none","l2")
            b_sim_cos_std = cos_sim_func(b_pred, b_targ)
            sim_cos_std.append(b_sim_cos_std)
    sim_d = {
        "sim_obj": th.cat(sim_obj,dim=0),
        "loss_obj": th.cat(loss_obj,dim=0)
    }
    if auxiliary:
        sim_d["sim_cos_std"] = th.cat(sim_cos_std,dim=0)
    # do merged second
    if merge_group:
        un_group_id = th.unique(group_id)
        # batching
        m_num_batches = len(un_group_id) // m_batch_size + int(len(un_group_id) % m_batch_size != 0)
        m_sim_obj, m_loss_obj, m_sim_cos_std, m_group_id, m_mol_id = [], [], [], [], []
        for b in get_pbar(range(m_num_batches),run_d,desc="> merged metrics"):
            b_group_id = un_group_id[b*m_batch_size:(b+1)*m_batch_size]
            b_mask = th.isin(group_id,b_group_id)
            b_group_id = group_id[b_mask]
            b_mol_id = mol_id[b_mask]
            b_pred = pred[b_mask]
            b_targ = targ[b_mask]
            b_m_pred, b_m_targ, b_m_mol_id, b_m_group_id = merge_group_func(
                b_pred, b_targ, b_group_id, b_mol_id, "obj"
            )
            b_m_sim_obj = obj_sim_func(b_m_pred, b_m_targ)
            b_m_loss_obj = obj_loss_func(b_m_pred, b_m_targ)
            m_sim_obj.append(b_m_sim_obj)
            m_loss_obj.append(b_m_loss_obj)
            m_group_id.append(b_m_group_id)
            m_mol_id.append(b_m_mol_id)
            if auxiliary:
                # just doing cos, forget about the other ones
                b_pred = process_spec(unprocess_spec(b_pred, data_d["transform"]),"none","l2")
                b_targ = process_spec(unprocess_spec(b_targ, data_d["transform"]),"none","l2")
                b_m_pred, b_m_targ, b_m_mol_id, b_m_group_id = merge_group_func(
                    b_pred, b_targ, b_group_id, b_mol_id, "std"
                )
                b_m_sim_cos_std = cos_sim_func(b_m_pred, b_m_targ)
                m_sim_cos_std.append(b_m_sim_cos_std)
        m_group_id = th.cat(m_group_id,dim=0)
        m_mol_id = th.cat(m_mol_id,dim=0)
        sim_d["m_sim_obj"] = th.cat(m_sim_obj,dim=0)
        sim_d["m_loss_obj"] = th.cat(m_loss_obj,dim=0)
        sim_d["m_group_id"] = m_group_id
        sim_d["m_mol_id"] = m_mol_id
        if auxiliary:
            sim_d["m_sim_cos_std"] = th.cat(m_sim_cos_std,dim=0)
    # construct tables and compute metrics
    merged_flags = [False]
    if merge_group:
        merged_flags.append(True)
    groupby_mol_flags = [False]
    if groupby_mol:
        groupby_mol_flags.append(True)
    tables = []
    for sl in ["sim", "loss"]:
        for merged in merged_flags:
            keys, vals = [], []
            if merged:
                _mol_id = m_mol_id
                _group_id = m_group_id
                for k, v in sim_d.items():
                    if k.startswith(f"m_{sl}"):
                        keys.append(k[len(f"m_{sl}_"):])
                        vals.append(v)
            else:
                _mol_id = mol_id
                _group_id = group_id
                for k, v in sim_d.items():
                    if k.startswith(sl):
                        keys.append(k[len(f"{sl}_"):])
                        vals.append(v)
            # print(sl,keys)
            table = MetricTable(
                keys, vals, _mol_id, _group_id, prefix, loss=(
                    sl == "loss"), merged=merged)
            # compute all of the metrics
            for gm in groupby_mol_flags:
                table.compute(
                    compute_agg=compute_agg,
                    compute_hist=compute_hist,
                    groupby_mol=gm)
            tables.append(table)
    return tables


def run_val(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        use_wandb):

    if not (dl_d["primary"]["val"] is None):
        # stuff related to device
        dev = th.device(run_d["device"])
        nb = run_d["non_blocking"]
        # validation
        model.eval()
        pred, targ, mol_id, group_id = [], [], [], []
        with th.no_grad():
            for b_idx, b in get_pbar(
                enumerate(
                    dl_d["primary"]["val"]), run_d, desc="> val", total=len(
                    dl_d["primary"]["val"])):
                b = data_to_device(b, dev, nb)
                b_pred = model(data=b, amp=run_d["amp"])["pred"]
                b_targ = b["spec"]
                b_mol_id = b["mol_id"]
                b_group_id = b["group_id"]
                pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                mol_id.append(b_mol_id.detach().to("cpu", non_blocking=nb))
                group_id.append(b_group_id.detach().to("cpu", non_blocking=nb))
        pred = th.cat(pred, dim=0)
        targ = th.cat(targ, dim=0)
        mol_id = th.cat(mol_id, dim=0)
        group_id = th.cat(group_id, dim=0)
        tables = compute_metric_tables(
            pred, targ, mol_id, group_id, "val", 
            data_d, run_d,
            auxiliary=run_d["log_auxiliary"],
            merge_group=True,
            compute_agg=True,
            compute_hist=False,
            groupby_mol=True
        )
        # print("val",lsh_d.keys())
        out_d = {}
        for table in tables:
            out_d = dict(
                **out_d,
                **table.unload_cache(prefix=False, agg=True, hist=False),
                **table.export_val("obj")
            )
        stop_key = run_d["stop_key"]
        spec_loss_obj_mean = out_d["spec_loss_obj_mean"]
        mol_loss_obj_mean = out_d["mol_loss_obj_mean"]
        loss_mean = out_d[stop_key]
        print(f"> step {step}, epoch {epoch}: val, {stop_key}: {loss_mean:.4f}")
        log_d = {"epoch": epoch, "Epoch": epoch}
        for table in tables:
            for k, v in table.unload_cache(
                    agg=True, hist=run_d["save_media"]).items():
                log_d[k] = v
        if use_wandb:
            wandb.log(log_d, commit=False)
            wandb.log({}, commit=True)
        if run_d["print_stats"]:
            pprint(log_d)
    else:
        out_d = {run_d["stop_key"]: np.nan}
    return step, epoch, out_d



def run_test(
        step,
        epoch,
        model,
        dl_d,
        data_d,
        model_d,
        run_d,
        use_wandb,
        run_dir,
        test_sets=None):

    if test_sets is None:
        test_sets = ["test"]
    if run_d["do_test"]:
        # stuff related to device
        dev = th.device(run_d["device"])
        nb = run_d["non_blocking"]
        print(">> test")
        # model setup
        model.to(dev)
        model.eval()
        out_d, save_tables = {}, []
        for order in ["primary", "secondary"]:
            out_d[order] = {}
            for dl_key, dl in dl_d[order].items():
                if not (dl_key in test_sets) or dl is None:
                    continue
                pred, targ, mol_id, group_id = [], [], [], []
                with th.no_grad():
                    for b_idx, b in get_pbar(
                            enumerate(dl), run_d, desc=f"> {dl_key}", total=len(dl)):
                        b = data_to_device(b, dev, nb)
                        b_pred = model(data=b, amp=run_d["amp"])["pred"]
                        b_targ = b["spec"]
                        b_mol_id = b["mol_id"]
                        b_group_id = b["group_id"]
                        pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                        targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                        mol_id.append(
                            b_mol_id.detach().to(
                                "cpu", non_blocking=nb))
                        group_id.append(
                            b_group_id.detach().to(
                                "cpu", non_blocking=nb))
                pred = th.cat(pred, dim=0)
                targ = th.cat(targ, dim=0)
                mol_id = th.cat(mol_id, dim=0)
                group_id = th.cat(group_id, dim=0)
                tables = compute_metric_tables(
                    pred, targ, mol_id, group_id, dl_key,
                    data_d, run_d,
                    auxiliary=run_d["log_auxiliary"],
                    merge_group=True,
                    compute_agg=True,
                    compute_hist=run_d["save_media"],
                    groupby_mol=True
                )
                _out_d = {}
                for table in tables:
                    _out_d = dict(
                        **_out_d, **table.unload_cache(prefix=False, agg=True, hist=False))
                stop_key = run_d["stop_key"]
                spec_loss_obj_mean = _out_d["spec_loss_obj_mean"]
                mol_loss_obj_mean = _out_d["mol_loss_obj_mean"]
                loss_mean = _out_d[stop_key]
                print(
                    f"> {dl_key}, {stop_key} = {loss_mean:.4}")
                out_d[order] = _out_d
                log_d = {"epoch": epoch, "Epoch": epoch}
                for table in tables:
                    for k, v in table.unload_cache(
                            hist=run_d["save_media"]).items():
                        log_d[k] = v
                if use_wandb:
                    wandb.log(log_d, commit=False)
                if run_d["print_stats"]:
                    pprint(log_d)
                if run_d["save_test_sims"]:
                    # save the tables
                    save_tables.extend(tables)
        if run_d["save_test_sims"]:
            save_dp = os.path.join(run_dir, "save_tables")
            os.makedirs(save_dp, exist_ok=True)
            for table in save_tables:
                save_str = table.get_table_str()
                save_fp = os.path.join(save_dp, save_str)
                table.save(save_fp)
                if use_wandb:
                    wandb.save(save_fp, base_path=run_dir)
        if use_wandb:
            wandb.log({}, commit=True)
    else:
        out_d = {}
    return step, epoch, out_d



def train_and_eval(data_d, model_d, run_d, use_wandb):

    # set seeds
    th.manual_seed(run_d["train_seed"])
    np.random.seed(run_d["train_seed"] // 2)

    # set parallel strategy
    if run_d["parallel_strategy"] == "fd":
        parallel_strategy = "file_descriptor"
    else:
        parallel_strategy = "file_system"
    th.multiprocessing.set_sharing_strategy(parallel_strategy)

    # set determinism (this seems to only affect CNN)
    if run_d["cuda_deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    th.use_deterministic_algorithms(run_d["cuda_deterministic"])

    # load dataset, set up model
    ds, model, casmi_ds, pcasmi_ds, casmi22_ds = get_ds_model(data_d, model_d, run_d)
    num_params = count_parameters(model, requires_grad=False)
    mol_embed_params, mlp_params, total_params = model.count_parameters()
    assert num_params == total_params, (num_params, total_params)
    print(f">>> mol_embed_params = {mol_embed_params}, mlp_params = {mlp_params}, total_params = {total_params}")

    if run_d["dp"]:
        assert run_d["device"] == "cuda:0"
        assert run_d["dp_num_gpus"] > 1
        model = th.nn.DataParallel(
            model, device_ids=[
                i for i in range(
                    run_d["dp_num_gpus"])])

    # set up dataloader
    dl_d, split_id_d = ds.get_dataloaders(run_d)

    # set up optimizer
    if run_d["optimizer"] == "adam":
        optimizer_fn = th.optim.Adam
    elif run_d["optimizer"] == "adamw":
        optimizer_fn = th.optim.AdamW
    else:
        raise NotImplementedError
    if run_d["pt_weight_decay"] == -1.:
        # use the same amount of weight decay for everything
        optimizer = optimizer_fn(
            model.parameters(),
            lr=run_d["learning_rate"],
            weight_decay=run_d["weight_decay"])
    else:
        # use different weight decay for pretrained part
        # this only works for pretrained models
        if run_d["dp"]:
            # dataparallel
            nopt_params, pt_params = model.module.get_split_params()
        else:
            nopt_params, pt_params = model.get_split_params()
        optimizer = optimizer_fn(
            [
                {"params": nopt_params, "weight_decay": run_d["weight_decay"]},
                {"params": pt_params, "weight_decay": run_d["pt_weight_decay"]}
            ],
            lr=run_d["learning_rate"]
        )

    # set up scheduler
    if run_d["scheduler"] == "step":
        scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            run_d["scheduler_period"],
            gamma=run_d["scheduler_ratio"]
        )
    elif run_d["scheduler"] == "plateau":
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=run_d["scheduler_period"],
            factor=run_d["scheduler_ratio"]
        )
    elif run_d["scheduler"] == "polynomial":
        # special kind of training for Graphormer
        # note: ignores the learning rate/weight decay stuff passed to
        # optimizer
        if run_d["num_decay_epochs"] == -1:
            num_decay_epochs = run_d["num_epochs"]
        else:
            num_decay_epochs = run_d["num_decay_epochs"]
            if run_d["num_decay_epochs"] > run_d["num_epochs"]:
                print(
                    f">>> WARNING: num_decay_epochs ({run_d['num_decay_epochs']}) > num_epochs ({run_d['num_epochs']})")
        if dl_d["primary"]["train"] is None:
            num_batches = 0
        else:
            num_batches = len(dl_d["primary"]["train"])
        tot_updates = num_decay_epochs * \
            (num_batches // run_d["grad_acc_interval"])
        warmup_updates = int(run_d["scheduler_warmup_frac"] * tot_updates)
        peak_lr = run_d["scheduler_peak_lr"]
        end_lr = run_d["scheduler_end_lr"]
        scheduler = PolynomialDecayLR(
            optimizer,
            warmup_updates=warmup_updates,  # warmup
            tot_updates=tot_updates,  # total
            lr=peak_lr,  # high
            end_lr=end_lr,  # low
            power=run_d["scheduler_power"]  # power
        )
    elif run_d["scheduler"] == "none":
        scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            1,
            gamma=1.0
        )
    else:
        raise NotImplementedError

    # load saved model from checkpoint
    if model_d["checkpoint_name"] is not None:
        chkpt_fp = os.path.join(
            data_d["checkpoint_dp"],
            model_d["checkpoint_name"] + ".pkl")
        chkpt_d = th.load(chkpt_fp,map_location="cpu")
        model.load_state_dict(chkpt_d["best_model_sd"])

    best_val_loss_mean = np.inf
    best_val_metrics = {}
    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())
    early_stop_count = 0
    early_stop_thresh = run_d["early_stop_thresh"]
    step = 0
    epoch = -1
    # casmi_d = init_casmi_d()
    # pcasmi_d = init_casmi_d()
    # casmi22_d = init_casmi_d()
    dev = th.device(run_d["device"])

    mr_fp = os.path.join(run_dir, "chkpt.pkl")
    temp_mr_fp = os.path.join(run_dir, "temp_chkpt.pkl")
    split_id_fp = os.path.join(run_dir, "split_id.pkl")
    if os.path.isfile(mr_fp):
        print(">>> reloading model from most recent checkpoint")
        mr_d = th.load(mr_fp,map_location="cpu")
        model.load_state_dict(mr_d["mr_model_sd"])
        best_state_dict = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(mr_d["optimizer_sd"])
        scheduler.load_state_dict(mr_d["scheduler_sd"])
        best_val_loss_mean = mr_d["best_val_loss_mean"]
        best_val_metrics = mr_d["best_val_metrics"]
        best_epoch = mr_d["best_epoch"]
        early_stop_count = mr_d["early_stop_count"]
        step = mr_d["step"]
        epoch = mr_d["epoch"]
        casmi_d = mr_d["casmi_d"]
        pcasmi_d = mr_d["pcasmi_d"]
        casmi22_d = mr_d["casmi22_d"]
    else:
        print(">>> no checkpoint detected")
        mr_d = {
            "mr_model_sd": model.state_dict(),
            "best_model_sd": best_state_dict,
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_val_loss_mean": best_val_loss_mean,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,

        }
        if run_d["save_split"]:
            # save data split
            th.save(split_id_d, split_id_fp)
            if use_wandb:
                wandb.save("split_id.pkl")
        if run_d["save_state"]:
            # save model state
            th.save(mr_d,temp_mr_fp)
            os.replace(temp_mr_fp,mr_fp)
            if use_wandb:
                wandb.save("chkpt.pkl")
    model.to(dev)

    epoch += 1

    while epoch < run_d["num_epochs"]:

        print(f">>> start epoch {epoch}")

        # training, single epoch
        step, epoch, train_d = run_train_epoch(
            step, epoch, model, dl_d, data_d, model_d, run_d, use_wandb, optimizer, scheduler)

        # validation
        step, epoch, val_d = run_val(
            step, epoch, model, dl_d, data_d, model_d, run_d, use_wandb)

        # update scheduler
        if run_d["scheduler"] == "step":
            scheduler.step()
        elif run_d["scheduler"] == "plateau":
            scheduler.step(val_d[run_d["stop_key"]])

        # tracking
        step, epoch, track_d = run_track(
            step, epoch, model, dl_d, data_d, model_d, run_d, use_wandb, ds, val_d)

        # early stopping
        val_loss_mean = val_d[run_d["stop_key"]]
        if best_val_loss_mean == np.inf:
            print(f"> val loss delta: N/A")
        else:
            print(f"> val loss delta: {val_loss_mean-best_val_loss_mean}")
        if run_d["use_val_info"]:
            if best_val_loss_mean < val_loss_mean:
                early_stop_count += 1
                print(
                    f"> val loss DID NOT decrease, early stop count at {early_stop_count}/{early_stop_thresh}")
            else:
                best_val_loss_mean = val_loss_mean
                best_val_metrics = {
                    k: v for k, v in val_d.items() if (
                        "_mean" in k)}
                best_epoch = epoch
                early_stop_count = 0
                # update state dicts
                model.to("cpu")
                best_state_dict = copy.deepcopy(model.state_dict())
                model.to(dev)
                print("> val loss DID decrease, early stop count reset")
            if early_stop_count == early_stop_thresh:
                print("> early stopping NOW")
                break
        else:
            # always assume the most recent epoch is the best
            best_val_loss_mean = val_loss_mean
            best_val_metrics = {
                k: v for k,
                v in val_d.items() if (
                    "_mean" in k)}
            best_epoch = epoch
            early_stop_count = 0
            # update state dicts
            model.to("cpu")
            best_state_dict = copy.deepcopy(model.state_dict())
            model.to(dev)

        # save model
        mr_d = {
            "mr_model_sd": model.state_dict(),
            "best_model_sd": best_state_dict,
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_val_loss_mean": best_val_loss_mean,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "casmi": False,
            "pcasmi": False,
            "casmi22": False,
            "casmi_d": casmi_d,
            "pcasmi_d": pcasmi_d,
            "casmi22_d": casmi22_d
        }
        if run_d["save_state"]:
            th.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp,mr_fp)
            if use_wandb:
                wandb.save("chkpt.pkl")
        if use_wandb:
            # sync wandb (after epoch is complete!)
            wandb.log({"commit": epoch}, commit=True)

        epoch += 1
    
    def update_mr_d(mr_d,**kwargs):

        for k, v in kwargs.items():
            mr_d[k] = v
        if run_d["save_state"]:
            th.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp, mr_fp)
            if use_wandb:
                wandb.save("chkpt.pkl")

    # test
    if not mr_d["test"]:
        compute_cross_sims(data_d, run_d, dl_d, use_wandb)
        model.load_state_dict(best_state_dict)
        step, epoch, test_d = run_test(step, epoch, model, dl_d, data_d,
                                    model_d, run_d, use_wandb, run_dir, test_sets=run_d["test_sets"])
        update_mr_d(mr_d,test=True)

    # get memory usage
    if run_d["device"] != "cpu" and th.cuda.is_available():
        cuda_max_memory = th.cuda.max_memory_allocated(device=dev)/1e9
        print(f"> GPU memory: {cuda_max_memory:.2f} GB")

    return