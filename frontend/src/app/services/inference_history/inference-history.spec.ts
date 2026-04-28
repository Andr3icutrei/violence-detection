import { TestBed } from '@angular/core/testing';

import { InferenceHistoryService } from './inference-history.service';

describe('InferenceHistory', () => {
  let service: InferenceHistoryService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(InferenceHistoryService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
