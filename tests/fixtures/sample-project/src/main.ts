// Main entry point for the sample application
// Coordinates auth and form modules

import { verifyToken, AuthService } from './auth';
import { createForm } from './forms/createForm';
import { validateForm } from './forms/validateForm';

export async function main() {
  const auth = new AuthService();
  const token = await auth.login('user', 'pass');

  if (verifyToken(token)) {
    const form = createForm({ title: 'Patient Intake' });
    const result = validateForm(form);
    console.log('Form valid:', result.valid);
  }
}

main().catch(console.error);
