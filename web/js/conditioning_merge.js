// Conditioning Merge frontend: hide widgets that don't apply to the selected mode.

const { app } = window.comfyAPI.app;

// Node type names registered in conditioning_rebalance.py
const MERGE_NODES = ["ConditioningMerge", "ConditioningMergeMulti", "ConditioningMergeList"];

// Widgets that are only relevant for specific modes.
const HIDEABLE_WIDGETS = ["match_percent", "strength", "metric", "target_tokens", "distribute"];

// Which of the hideable widgets apply to each merge mode.
const MODE_WIDGETS = {
    top_match: ["match_percent"],
    average: [],
    norm_average: [],
    weighted: ["strength"],
    norm_weighted: ["strength"],
    add: [],
    subtract: [],
    max_magnitude: [],
    min_magnitude: [],
    concat: ["strength", "metric", "target_tokens", "distribute"],
    difference: ["strength"],
    orthogonal: ["strength"],
};

// Qwen VL Image Embeds Merge: batch style reducers. Strength only applies
// to a subset of modes; it is hidden for the rest.
const QWEN_MERGE_NODES = ["QwenVLImageEmbedsMerge"];

const QWEN_HIDEABLE_WIDGETS = ["strength", "harmonic_sign"];

const QWEN_MODE_WIDGETS = {
    mean: [],
    median: [],
    trimmed_mean: ["strength"],
    consensus: ["strength"],
    dominant: [],
    spread: ["strength"],
    centroid_dir: [],
    average: [],
    norm_average: [],
    weighted: ["strength"],
    norm_weighted: ["strength"],
    add: [],
    subtract: [],
    max_magnitude: [],
    min_magnitude: [],
    top_match: ["strength"],
    concat: [],
    geometric_mean: [],
    harmonic_mean: ["harmonic_sign"],
    rms: [],
    mode_stats: [],
    pca_style: [],
    outlier: ["strength"],
    random: ["strength"],
};

function showWidget(w) {
    if (w._mergeOrigType != null) {
        w.type = w._mergeOrigType;
        w._mergeOrigType = null;
    }
    delete w.computeSize;
    w.hidden = false;
}

function hideWidget(w) {
    if (w._mergeOrigType == null) {
        w._mergeOrigType = w.type;
    }
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
}

function applyModeVisibility(node) {
    if (!node || !node.widgets) return;

    const isQwen = QWEN_MERGE_NODES.includes(node.comfyClass || node.constructor?.comfyClass || node.getTitle?.());
    const modeWidget = node.widgets.find((w) => w.name === "mode");
    const mode = modeWidget ? String(modeWidget.value) : "top_match";
    const table = isQwen ? QWEN_MODE_WIDGETS : MODE_WIDGETS;
    const hideable = isQwen ? QWEN_HIDEABLE_WIDGETS : HIDEABLE_WIDGETS;
    const visible = new Set(table[mode] || []);

    let changed = false;
    for (const w of node.widgets) {
        if (!hideable.includes(w.name)) continue;
        const shouldShow = visible.has(w.name);
        const isHidden = w.type === "hidden";
        if (shouldShow && isHidden) {
            showWidget(w);
            changed = true;
        } else if (!shouldShow && !isHidden) {
            hideWidget(w);
            changed = true;
        }
    }

    if (changed && typeof node.setSize === "function" && typeof node.computeSize === "function") {
        node.setSize(node.computeSize());
    }
}

function hookModeWidget(node) {
    const modeWidget = node.widgets && node.widgets.find((w) => w.name === "mode");
    if (!modeWidget || modeWidget._mergeHooked) return;
    modeWidget._mergeHooked = true;

    const origCallback = modeWidget.callback;
    modeWidget.callback = function (...args) {
        const ret = origCallback ? origCallback.apply(this, args) : undefined;
        applyModeVisibility(node);
        return ret;
    };
}

app.registerExtension({
    name: "Rebalance.ConditioningMergeMode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!MERGE_NODES.includes(nodeData?.name) && !QWEN_MERGE_NODES.includes(nodeData?.name)) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            hookModeWidget(this);
            applyModeVisibility(this);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            // Widget values are restored during configure; re-sync visibility.
            hookModeWidget(this);
            applyModeVisibility(this);
            return r;
        };
    },
});
