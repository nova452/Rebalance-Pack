// Display frontend: renders the node's `ui.text` result inside a
// read-only text widget on the node itself (the core frontend only does
// this for its own built-in PreviewAny node, matched by class name).

const { app } = window.comfyAPI.app;
const { ComfyWidgets } = window.comfyAPI.widgets;

const DISPLAY_NAME = "Display";

app.registerExtension({
    name: "Rebalance-Pack.Display",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== DISPLAY_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;

            // Placeholder widget updated on execution.
            this._displayWidget = ComfyWidgets["STRING"](
                this,
                "display",
                ["", { default: "", multiline: true, serialized: false }],
                app
            ).widget;
            this._displayWidget.value = "None";
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted
                ? onExecuted.apply(this, arguments)
                : undefined;

            if (message?.text?.length && this._displayWidget) {
                this._displayWidget.value = message.text[0];
                this.setDirtyCanvas(true, true);
            }
            return r;
        };
    },
});
