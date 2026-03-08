// Form validation module
// Validates form structure and field requirements

import { Form, FormField } from './createForm';

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
}

export function validateForm(form: Form): ValidationResult {
  const errors: ValidationError[] = [];

  if (!form.title || form.title.trim().length === 0) {
    errors.push({ field: 'title', message: 'Form title is required' });
  }

  for (const field of form.fields) {
    if (!field.name) {
      errors.push({ field: 'fields', message: 'Field name is required' });
    }
    if (!['text', 'number', 'date', 'select'].includes(field.type)) {
      errors.push({ field: field.name, message: `Invalid field type: ${field.type}` });
    }
  }

  return { valid: errors.length === 0, errors };
}

export function validateField(field: FormField, value: unknown): ValidationError | null {
  if (field.required && (value === null || value === undefined || value === '')) {
    return { field: field.name, message: `${field.label ?? field.name} is required` };
  }

  if (field.type === 'number' && typeof value === 'string' && isNaN(Number(value))) {
    return { field: field.name, message: `${field.label ?? field.name} must be a number` };
  }

  return null;
}
