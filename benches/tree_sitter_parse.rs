//! Benchmark: Native tree-sitter parsing across 10 languages.
//!
//! Measures parse + symbol extraction time per language.
//! TS baseline: 5-20ms (WASM). Rust target: 1-5ms (native C grammars).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use contextplus_rs::core::tree_sitter::parse_with_tree_sitter;

/// Realistic code samples for each supported language.
fn code_samples() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "typescript",
            "ts",
            r#"
import { Injectable } from '@nestjs/common';
import { PrismaClient, Prisma } from '@prisma/client';

export interface ProfileServiceDependencies {
    prisma: PrismaClient;
    logger: Logger;
}

export function createProfileService(deps: ProfileServiceDependencies) {
    const { prisma, logger } = deps;

    async function findById(id: string): Promise<Profile | null> {
        logger.info({ id }, 'Finding profile by ID');
        return prisma.profile.findUnique({ where: { id } });
    }

    async function create(data: CreateProfileInput): Promise<Profile> {
        logger.info({ data }, 'Creating profile');
        const profile = await prisma.profile.create({ data });
        return profile;
    }

    async function update(id: string, data: UpdateProfileInput): Promise<Profile> {
        return prisma.profile.update({ where: { id }, data });
    }

    async function archive(id: string): Promise<void> {
        await prisma.profile.update({
            where: { id },
            data: { archivedAt: new Date() },
        });
    }

    return { findById, create, update, archive };
}

export type ProfileService = ReturnType<typeof createProfileService>;
"#,
        ),
        (
            "tsx",
            "tsx",
            r#"
import React, { useState, useEffect, useCallback } from 'react';

interface FormProps {
    onSubmit: (data: FormData) => Promise<void>;
    initialValues?: Partial<FormData>;
}

export const ContactForm: React.FC<FormProps> = ({ onSubmit, initialValues }) => {
    const [name, setName] = useState(initialValues?.name ?? '');
    const [email, setEmail] = useState(initialValues?.email ?? '');
    const [errors, setErrors] = useState<Record<string, string>>({});

    const validate = useCallback(() => {
        const newErrors: Record<string, string> = {};
        if (!name.trim()) newErrors.name = 'Name is required';
        if (!email.includes('@')) newErrors.email = 'Invalid email';
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    }, [name, email]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (validate()) {
            await onSubmit({ name, email });
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <input value={name} onChange={e => setName(e.target.value)} />
            <input value={email} onChange={e => setEmail(e.target.value)} />
            <button type="submit">Submit</button>
        </form>
    );
};
"#,
        ),
        (
            "javascript",
            "js",
            r#"
const express = require('express');
const router = express.Router();

function createAuthMiddleware(secretKey) {
    return function authenticate(req, res, next) {
        const token = req.headers.authorization?.split(' ')[1];
        if (!token) return res.status(401).json({ error: 'Missing token' });
        try {
            req.user = verify(token, secretKey);
            next();
        } catch (err) {
            return res.status(403).json({ error: 'Invalid token' });
        }
    };
}

class RateLimiter {
    constructor(maxRequests, windowMs) {
        this.maxRequests = maxRequests;
        this.windowMs = windowMs;
        this.requests = new Map();
    }

    check(ip) {
        const now = Date.now();
        const entry = this.requests.get(ip) || { count: 0, resetAt: now + this.windowMs };
        if (now > entry.resetAt) entry.count = 0;
        entry.count++;
        this.requests.set(ip, entry);
        return entry.count <= this.maxRequests;
    }
}

module.exports = { createAuthMiddleware, RateLimiter };
"#,
        ),
        (
            "python",
            "py",
            r#"
from dataclasses import dataclass
from typing import Optional, List
import asyncio

@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

class CacheManager:
    def __init__(self, max_size: int = 1000):
        self._cache: dict = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

def hash_content(text: str) -> str:
    h = 0
    for ch in text:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        if h > 0x7FFFFFFF:
            h -= 0x100000000
    return base36(h)

async def process_batch(items: List[str]) -> List[str]:
    tasks = [asyncio.create_task(process_item(item)) for item in items]
    return await asyncio.gather(*tasks)
"#,
        ),
        (
            "rust",
            "rs",
            r#"
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct VectorStore {
    dims: u32,
    count: u32,
    vectors: Vec<f32>,
    keys: Vec<String>,
    key_index: HashMap<String, usize>,
}

impl VectorStore {
    pub fn new(dims: u32, keys: Vec<String>, vectors: Vec<f32>) -> Self {
        let count = keys.len() as u32;
        let mut key_index = HashMap::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            key_index.insert(key.clone(), i);
        }
        Self { dims, count, vectors, keys, key_index }
    }

    pub fn find_nearest(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.count as usize);
        for i in 0..self.count as usize {
            let offset = i * self.dims as usize;
            let stored = &self.vectors[offset..offset + self.dims as usize];
            scored.push((i, cosine(query, stored)));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        scored.into_iter().map(|(i, s)| (self.keys[i].clone(), s)).collect()
    }
}

pub enum Error {
    Io(std::io::Error),
    Parse(String),
    NotFound(String),
}

pub trait Repository: Send + Sync {
    fn get(&self, id: &str) -> Result<Option<String>, Error>;
    fn put(&self, id: &str, value: &str) -> Result<(), Error>;
}
"#,
        ),
        (
            "go",
            "go",
            r#"
package server

import (
    "context"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type Server struct {
    Port     int
    Host     string
    mux      *http.ServeMux
    mu       sync.RWMutex
    handlers map[string]http.Handler
}

func NewServer(host string, port int) *Server {
    return &Server{
        Host:     host,
        Port:     port,
        mux:      http.NewServeMux(),
        handlers: make(map[string]http.Handler),
    }
}

func (s *Server) Start(ctx context.Context) error {
    addr := fmt.Sprintf("%s:%d", s.Host, s.Port)
    srv := &http.Server{Addr: addr, Handler: s.mux}

    go func() {
        <-ctx.Done()
        shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer cancel()
        srv.Shutdown(shutdownCtx)
    }()

    return srv.ListenAndServe()
}

type HealthChecker interface {
    Check(ctx context.Context) error
}
"#,
        ),
        (
            "java",
            "java",
            r#"
package com.example.service;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

public class UserService {
    private final ConcurrentHashMap<String, User> cache;
    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
        this.cache = new ConcurrentHashMap<>();
    }

    public Optional<User> findById(String id) {
        User cached = cache.get(id);
        if (cached != null) return Optional.of(cached);
        return repository.findById(id).map(user -> {
            cache.put(id, user);
            return user;
        });
    }

    public User create(CreateUserRequest request) {
        User user = new User(request.getName(), request.getEmail());
        repository.save(user);
        cache.put(user.getId(), user);
        return user;
    }
}

interface UserRepository {
    Optional<User> findById(String id);
    void save(User user);
    List<User> findAll();
}

enum UserRole {
    ADMIN,
    USER,
    MODERATOR
}
"#,
        ),
        (
            "c",
            "c",
            r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Vector {
    float *data;
    size_t len;
    size_t capacity;
};

struct Vector *vector_new(size_t capacity) {
    struct Vector *v = malloc(sizeof(struct Vector));
    v->data = malloc(capacity * sizeof(float));
    v->len = 0;
    v->capacity = capacity;
    return v;
}

void vector_push(struct Vector *v, float value) {
    if (v->len >= v->capacity) {
        v->capacity *= 2;
        v->data = realloc(v->data, v->capacity * sizeof(float));
    }
    v->data[v->len++] = value;
}

float vector_dot(const struct Vector *a, const struct Vector *b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a->len && i < b->len; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

void vector_free(struct Vector *v) {
    free(v->data);
    free(v);
}
"#,
        ),
        (
            "cpp",
            "cpp",
            r#"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
    virtual std::string name() const = 0;
};

class Circle : public Shape {
    double radius_;
public:
    explicit Circle(double radius) : radius_(radius) {}
    double area() const override { return 3.14159 * radius_ * radius_; }
    std::string name() const override { return "Circle"; }
};

struct Rectangle {
    double width;
    double height;
    double area() const { return width * height; }
};

template<typename T>
class Container {
    std::vector<T> items_;
public:
    void add(T item) { items_.push_back(std::move(item)); }
    size_t size() const { return items_.size(); }
    const T& get(size_t idx) const { return items_.at(idx); }
};
"#,
        ),
        (
            "bash",
            "sh",
            r#"
#!/usr/bin/env bash
set -euo pipefail

setup_env() {
    export NODE_ENV="${1:-development}"
    export PORT="${2:-3000}"
    echo "Environment: $NODE_ENV, Port: $PORT"
}

run_migrations() {
    local db_url="$1"
    echo "Running migrations against $db_url..."
    npx prisma migrate deploy
    echo "Migrations complete."
}

check_health() {
    local url="$1"
    local max_retries="${2:-10}"
    local count=0
    while [ $count -lt $max_retries ]; do
        if curl -sf "$url/health" > /dev/null 2>&1; then
            echo "Service is healthy"
            return 0
        fi
        count=$((count + 1))
        sleep 1
    done
    echo "Health check failed after $max_retries attempts"
    return 1
}

deploy() {
    setup_env "production"
    run_migrations "$DATABASE_URL"
    check_health "http://localhost:$PORT"
}
"#,
        ),
    ]
}

fn bench_parse_per_language(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_sitter_parse");
    let samples = code_samples();

    for (lang, ext, code) in &samples {
        group.bench_with_input(BenchmarkId::new("parse", *lang), lang, |bench, _| {
            bench.iter(|| {
                let symbols = parse_with_tree_sitter(code, ext).unwrap();
                assert!(!symbols.is_empty(), "no symbols parsed for {}", lang);
            });
        });
    }

    group.finish();
}

fn bench_parse_all_languages(c: &mut Criterion) {
    let samples = code_samples();

    c.bench_function("tree_sitter_parse_all_10", |bench| {
        bench.iter(|| {
            let mut total_symbols = 0;
            for (_, ext, code) in &samples {
                let symbols = parse_with_tree_sitter(code, ext).unwrap();
                total_symbols += symbols.len();
            }
            assert!(
                total_symbols > 20,
                "expected >20 symbols across 10 languages"
            );
            total_symbols
        });
    });
}

criterion_group!(benches, bench_parse_per_language, bench_parse_all_languages);
criterion_main!(benches);
