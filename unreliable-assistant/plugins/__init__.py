"""Lifecycle plugin system. See plugins/registry.py for the framework and plugins/builtin.py for the plugins."""
from plugins.builtin import CircuitBreaker, HitlGate, InjectionShield, OtelExporter, PiiRedactor, RagasScorer, TokenCounter
from plugins.registry import Block, PluginConfigError, PluginMeta, PluginRegistry, RequestContext, Stage, hookimpl

REQUIRED = ["pii_redactor", "injection_shield", "hitl_gate"]  # the service refuses to start without these


def default_factories(pii=None, injection=None, meter=None, embedder=None):
    return {"pii_redactor": lambda: PiiRedactor(pii), "injection_shield": lambda: InjectionShield(injection),
            "circuit_breaker": CircuitBreaker, "hitl_gate": HitlGate, "otel_exporter": lambda: OtelExporter(meter), "token_counter": lambda: TokenCounter(meter), "ragas_scorer": lambda: RagasScorer(embedder, meter)}


__all__ = ["Block", "CircuitBreaker", "HitlGate", "InjectionShield", "OtelExporter", "PiiRedactor", "PluginConfigError",
           "PluginMeta", "PluginRegistry", "REQUIRED", "RagasScorer", "RequestContext", "Stage", "TokenCounter", "default_factories", "hookimpl"]
