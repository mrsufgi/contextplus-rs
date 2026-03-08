# Architecture

## Overview

This project demonstrates a modular architecture with clear separation of concerns.

## Key Components

- [[src/auth]] — Authentication and authorization
- [[src/forms/createForm]] — Form builder with validation
- [[src/forms/validateForm]] — Input validation engine
- [[src/utils/helpers]] — Shared utilities

## Data Flow

1. User authenticates via `AuthService.login()`
2. Token verified with `verifyToken()`
3. Forms created via `createForm()`
4. Input validated with `validateForm()`
