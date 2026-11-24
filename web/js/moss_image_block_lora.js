// MOSSImageLoraLoaderBlock frontend: "Scan Blocks" reads the selected LoRA via
// /rebalance_pack/moss_image_lora_blocks and shows one 0.0-1.0 widget per block
// group found in the file. Groups with no keys in the file are hidden, and the
// result is stored in `scan_json` so the Python side knows which groups ran.

const { app } = window.comfyAPI.app;

const NODE_NAME = "MOSSImageLoraLoaderBlock";
const ROUTE = "/rebalance_pack/moss_image_lora_blocks";
const SCAN_WIDGET = "scan_json";

// Declared inputs that are not block weights (Widgets + the Scan button).
const FIXED_WIDGETS = new Set(["lora_name", "strength", SCAN_WIDGET, "Scan Blocks", "model"]);

// Mirror of krea2_block_lora.group_widget_name: group "layers.3." -> "layers_3".
function groupToWidget(group) {
    return String(group).replace(/\.+$/, "").replace(/\./g, "_");
}

function blockWidgetNames(nodeData) {
    const inputs = nodeData?.input?.required || {};
    return new Set(Object.keys(inputs).filter(name => !FIXED_WIDGETS.has(name)));
}

async function fetchBlocks(loraName) {
    const qs = new URLSearchParams({ lora_name: loraName });
    const res = await fetch(`${ROUTE}?${qs.toString()}`, { cache: "no-store" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    return Array.isArray(data.groups) ? data.groups : [];
}

function hideWidget(w) {
    if (w._origType == null) w._origType = w.type;
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
}

function showWidget(w) {
    if (w._origType != null) {
        w.type = w._origType;
        w._origType = null;
    }
    delete w.computeSize;
    w.hidden = false;
}

function setScan(node, groups) {
    const scan = node.widgets.find(w => w.name === SCAN_WIDGET);
    if (scan) {
        scan.value = JSON.stringify({ groups });
        if (scan.callback) scan.callback(scan.value);
    }
}

function getScannedGroups(node) {
    let data;
    try {
        data = JSON.parse(node.widgets.find(w => w.name === SCAN_WIDGET)?.value || "{}");
    } catch (e) { /* malformed scan_json */ }
    return Array.isArray(data?.groups) ? data.groups : null;
}

function applyScannedVisibility(node) {
    const groups = getScannedGroups(node);
    if (groups === null) return; // never scanned: show every block widget
    const present = new Set(groups.map(groupToWidget));
    for (const w of node.widgets) {
        if (!w._mossblock) continue;
        if (present.has(w.name)) { if (w.hidden) showWidget(w); }
        else if (!w.hidden) hideWidget(w);
    }
    if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
        node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
    }
}

async function scanBlocks(node) {
    const loraWidget = node.widgets.find(w => w.name === "lora_name");
    const btn = node.widgets.find(w => w.name === "Scan Blocks");
    const flash = (label, restore) => {
        if (!btn) return;
        btn.label = label;
        setTimeout(() => { btn.label = "Scan Blocks"; node.setDirtyCanvas(true, true); }, restore || 1200);
    };

    if (!loraWidget || !loraWidget.value) {
        flash("Pick a LoRA");
        node.setDirtyCanvas(true, true);
        return;
    }

    if (btn) { btn.disabled = true; btn.label = "Scanning..."; }
    node.setDirtyCanvas(true, true);
    try {
        const groups = await fetchBlocks(loraWidget.value);
        setScan(node, groups);
        applyScannedVisibility(node);
        flash(`Scanned (${groups.length})`, 2000);
    } catch (e) {
        console.error("[MOSSImageLoraLoaderBlock] scan failed:", e);
        flash("Scan failed");
    } finally {
        if (btn) btn.disabled = false;
        node.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Rebalance.MOSSImageLoraLoaderBlock",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const blockWidgets = blockWidgetNames(nodeData);

        const tagBlockWidgets = (node) => {
            for (const w of node.widgets || []) {
                if (blockWidgets.size) {
                    w._mossblock = blockWidgets.has(w.name);
                } else {
                    // Fallback if the node definition gives no input names.
                    w._mossblock = !FIXED_WIDGETS.has(w.name);
                }
            }
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            tagBlockWidgets(this);
            if (!this.widgets.some(w => w.name === "Scan Blocks")) {
                this.addWidget("button", "Scan Blocks", null, () => scanBlocks(this));
            }
            applyScannedVisibility(this);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            tagBlockWidgets(this);
            applyScannedVisibility(this);
            return r;
        };
    },
});
