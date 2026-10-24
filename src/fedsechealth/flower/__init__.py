"""Flower deployment of FedSecHealth: one SuperNode per hospital, one SuperLink server.

The research simulator (``fedsechealth.fl``) runs every hospital in one
process. This package runs the *same* hospital logic (``fl.Hospital``: local
training, optional DP-SGD with Opacus) as a real Flower ``ClientApp``, so each
hospital can live in its own process or container and talk to the server over
gRPC with TLS.
"""
