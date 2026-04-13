import { TestBed } from '@angular/core/testing';

import { InferenceActionsService } from './inference-actions.service';

describe('InferenceActions', () => {
  let service: InferenceActionsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(InferenceActionsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
