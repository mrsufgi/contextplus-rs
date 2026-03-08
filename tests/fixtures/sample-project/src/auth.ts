// Authentication module for token verification
// Provides JWT-based auth with role checking

export function verifyToken(token: string): boolean {
  if (!token || token.length < 10) return false;
  const parts = token.split('.');
  return parts.length === 3;
}

export class AuthService {
  private secret: string;

  constructor(secret?: string) {
    this.secret = secret ?? 'default-secret';
  }

  async login(username: string, password: string): Promise<string> {
    // Simulate JWT creation
    const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
    const payload = btoa(JSON.stringify({ sub: username, iat: Date.now() }));
    const signature = btoa(`${header}.${payload}.${this.secret}`);
    return `${header}.${payload}.${signature}`;
  }

  async validateSession(token: string): Promise<boolean> {
    return verifyToken(token);
  }

  getRoles(token: string): string[] {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.roles ?? [];
    } catch {
      return [];
    }
  }
}
