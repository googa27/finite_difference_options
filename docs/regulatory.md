# Regulatory Reporting

This module offers minimal structures for representing trades, risk factors and exposures.

## Assumptions

- Only a handful of fields are captured for each trade and risk factor.
- CRIF output includes trade identifier, risk factor name and exposure amount.
- CUSO, Basel and FRTB calculations are placeholders that return a "not implemented" status.

## Limitations

- The simplified data model does not cover product-specific attributes.
- No validation against official regulatory schemas is performed.
- Risk calculations are not implemented and must be provided by a risk engine.
