from .visual_flow_connector import VisualFlowConnector

def get_connector(config):
    """
    Factory function to get the connector class based on the type.
    Args:
        config: Model config with connector_config dict.
    Returns:
        Connector instance corresponding to the specified type.
    """
    connector_type = config.connector_config["connector_type"]

    if connector_type == "visual_flow":
        return VisualFlowConnector(
            lang_dim=config.hidden_size,
        )
    raise ValueError(f"Unknown connector type: {connector_type}; expected 'visual_flow'")
