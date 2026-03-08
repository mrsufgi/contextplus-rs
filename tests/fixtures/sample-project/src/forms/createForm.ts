// Form creation module
// Builds form objects with validation config

export interface FormConfig {
  title: string;
  fields?: FormField[];
  validateOnBlur?: boolean;
}

export interface FormField {
  name: string;
  type: 'text' | 'number' | 'date' | 'select';
  required?: boolean;
  label?: string;
}

export interface Form {
  id: string;
  title: string;
  fields: FormField[];
  config: FormConfig;
  createdAt: number;
}

export function createForm(config: FormConfig): Form {
  const defaultFields: FormField[] = [
    { name: 'firstName', type: 'text', required: true, label: 'First Name' },
    { name: 'lastName', type: 'text', required: true, label: 'Last Name' },
    { name: 'dob', type: 'date', required: true, label: 'Date of Birth' },
  ];

  return {
    id: `form-${Date.now()}`,
    title: config.title,
    fields: config.fields ?? defaultFields,
    config,
    createdAt: Date.now(),
  };
}

export function cloneForm(form: Form, newTitle?: string): Form {
  return {
    ...form,
    id: `form-${Date.now()}`,
    title: newTitle ?? `${form.title} (Copy)`,
    createdAt: Date.now(),
  };
}
