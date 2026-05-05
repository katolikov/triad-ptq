"""Export TRIAD checkpoints to deployment formats.

Currently provides MLC-compatible export (mlc.py) for Vulkan-on-Android
deployment. The final `mlc_llm compile` step requires the MLC-LLM Python
package and Android NDK; the export step itself only produces the input
artefacts (packed weights, manifest, config) and runs without those.
"""
