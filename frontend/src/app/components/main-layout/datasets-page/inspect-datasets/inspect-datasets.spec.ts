import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InspectDatasets } from './inspect-datasets';

describe('InspectDatasets', () => {
  let component: InspectDatasets;
  let fixture: ComponentFixture<InspectDatasets>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InspectDatasets],
    }).compileComponents();

    fixture = TestBed.createComponent(InspectDatasets);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
